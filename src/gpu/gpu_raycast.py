"""
GPU-accelerated raycast sensor computation for all cars simultaneously.

The wall map is uploaded to the GPU once (and again only if a different
array is passed). Every frame, all ``cars x 7`` rays are marched in parallel
by a small CUDA kernel, one thread per ray, replacing the per-car per-ray
pixel-walking loop in ``Car.check_sensor``.

Falls back to a vectorised NumPy implementation if CuPy/CUDA is unavailable.
Both paths return the same values: the ray stops at the first wall pixel or the
first step outside the track, or after ``MAX_RAY_STEPS`` steps.
"""

import numpy as np
from src.render.car import Car

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


# Sensor angles relative to the car's heading (degrees), shared with Car.
SENSOR_ANGLES_DEG = list(Car.SENSOR_ANGLES)
MAX_RAY_STEPS = Car.SENSORS_DRAW_DISTANCE


# ---------------------------------------------------------------------------
# CUDA kernel source (used when CuPy is available)
# ---------------------------------------------------------------------------

_RAYCAST_KERNEL_SOURCE = r"""
extern "C" __global__
void raycast_kernel(
    const unsigned char* track_pixels,  // H x W, non-zero = wall, 0 = drivable
    int track_w,
    int track_h,
    const float* cx,          // center_x per car  [N]
    const float* cy,          // center_y per car  [N]
    const float* cos_arr,     // cos(ray_angle)   [N * R]
    const float* sin_arr,     // sin(ray_angle)   [N * R]
    float* out,               // [3 * N * R]: distances, then hit x, then hit y
    int max_steps,
    int N,
    int R
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * R) return;

    int car_i = idx / R;
    float ox = cx[car_i];
    float oy = cy[car_i];
    float dx = cos_arr[idx];
    float dy = sin_arr[idx];

    float px = ox;
    float py = oy;
    int step;
    for (step = 1; step <= max_steps; step++) {
        px = ox + dx * step;
        py = oy + dy * step;
        int ix = (int)px;
        int iy = (int)py;
        if (ix < 0 || ix >= track_w || iy < 0 || iy >= track_h) break;
        if (track_pixels[iy * track_w + ix] != 0) break;
    }

    float dist = sqrtf((px - ox)*(px - ox) + (py - oy)*(py - oy));
    int total = N * R;
    out[idx]             = dist;
    out[total + idx]     = px;
    out[2 * total + idx] = py;
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None and _CUPY_AVAILABLE:
        module = cp.RawModule(code=_RAYCAST_KERNEL_SOURCE)
        _kernel = module.get_function("raycast_kernel")
    return _kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GPURaycast:
    """
    Computes sensor distances for all alive cars in a single GPU call.
    """

    def __init__(self, track_w: int, track_h: int):
        self.track_w = track_w
        self.track_h = track_h
        self._cached_track = None     # the host array last uploaded (held to make `is` safe)
        self._track_gpu = None
        self.use_gpu = _CUPY_AVAILABLE
        self._sensor_offsets = np.array(SENSOR_ANGLES_DEG, dtype=np.float32)

    def _update_track(self, track_pixels_np: np.ndarray):
        """Upload the track pixel array to the GPU if it changed.
        
        IMPORTANT: The array must be C-contiguous (row-major) because the CUDA
        kernel indexes with [iy * track_w + ix].
        """
        # Identity check. We keep a reference to the cached array, so its id can't be
        # recycled by a different array (which a bare id() comparison would allow).
        if self._cached_track is track_pixels_np:
            return
        self._cached_track = track_pixels_np

        if self.use_gpu:
            self._track_gpu = cp.asarray(np.ascontiguousarray(track_pixels_np))

    def compute(
        self,
        track_pixels_np: np.ndarray,        # (H, W) uint8; 255 = wall, 0 = drivable
        centers_x: np.ndarray,              # (N,) float32, only alive cars
        centers_y: np.ndarray,              # (N,) float32
        angles_deg: np.ndarray,             # (N,) float32, heading per car
    ):
        """
        Returns:
            distances: (N, R) float32 array of sensor distances
            hit_x:     (N, R) float32 hit-point x
            hit_y:     (N, R) float32 hit-point y
        """
        self._update_track(track_pixels_np)

        N = len(centers_x)
        R = len(SENSOR_ANGLES_DEG)
        if N == 0:
            empty = np.zeros((0, R), dtype=np.float32)
            return empty, empty.copy(), empty.copy()

        # Ray directions, (N*R,): angle = 360 - (heading + sensor_offset)  [car.py convention]
        ray_angles_deg = (360.0 - (angles_deg[:, None] + self._sensor_offsets[None, :]))  # (N,R)
        ray_angles_rad = np.deg2rad(ray_angles_deg).astype(np.float32)              # (N,R)

        cos_vals = np.cos(ray_angles_rad).reshape(-1).astype(np.float32)  # (N*R,)
        sin_vals = np.sin(ray_angles_rad).reshape(-1).astype(np.float32)

        if self.use_gpu:
            return self._compute_gpu(centers_x, centers_y, cos_vals, sin_vals, N, R)
        else:
            return self._compute_cpu(track_pixels_np, centers_x, centers_y, cos_vals, sin_vals, N, R)

    def _compute_gpu(self, cx, cy, cos_vals, sin_vals, N, R):
        kernel = _get_kernel()

        cx_gpu  = cp.asarray(cx, dtype=cp.float32)
        cy_gpu  = cp.asarray(cy, dtype=cp.float32)
        cos_gpu = cp.asarray(cos_vals)
        sin_gpu = cp.asarray(sin_vals)
        out_gpu = cp.empty(3 * N * R, dtype=cp.float32)   # every slot is written by the kernel

        threads = 256
        blocks = (N * R + threads - 1) // threads

        kernel(
            (blocks,), (threads,),
            (
                self._track_gpu,
                np.int32(self.track_w),
                np.int32(self.track_h),
                cx_gpu, cy_gpu,
                cos_gpu, sin_gpu,
                out_gpu,
                np.int32(MAX_RAY_STEPS),
                np.int32(N),
                np.int32(R),
            )
        )

        out = cp.asnumpy(out_gpu).reshape(3, N, R)   # one device->host transfer
        return out[0], out[1], out[2]

    # Rays processed per CPU chunk: bounds temporary memory to ~a few dozen MB
    # instead of allocating (N*R, MAX_RAY_STEPS) arrays for the whole population.
    CPU_CHUNK_RAYS = 1024

    def _compute_cpu(self, track_pixels_np, cx, cy, cos_vals, sin_vals, N, R):
        """NumPy CPU fallback — vectorised over rays and steps, chunked over rays."""
        track_w = self.track_w
        track_h = self.track_h
        steps = np.arange(1, MAX_RAY_STEPS + 1, dtype=np.float32)  # (S,)

        ox_all = np.repeat(cx, R).astype(np.float32)  # (N*R,)
        oy_all = np.repeat(cy, R).astype(np.float32)

        total = N * R
        dist = np.empty(total, dtype=np.float32)
        hx = np.empty(total, dtype=np.float32)
        hy = np.empty(total, dtype=np.float32)

        for lo in range(0, total, self.CPU_CHUNK_RAYS):
            hi = min(lo + self.CPU_CHUNK_RAYS, total)
            ox, oy = ox_all[lo:hi], oy_all[lo:hi]

            px = ox[:, None] + cos_vals[lo:hi, None] * steps[None, :]   # (C, S)
            py = oy[:, None] + sin_vals[lo:hi, None] * steps[None, :]

            oob = (px < 0) | (px >= track_w) | (py < 0) | (py >= track_h)
            ix = np.clip(px.astype(np.int32), 0, track_w - 1)
            iy = np.clip(py.astype(np.int32), 0, track_h - 1)
            hit_mask = oob | (track_pixels_np[iy, ix] != 0)

            # First hit step; if the ray never hits, use the last step.
            hit_idx = np.argmax(hit_mask, axis=1)
            hit_idx[~hit_mask.any(axis=1)] = MAX_RAY_STEPS - 1

            ray_idx = np.arange(hi - lo)
            chx = px[ray_idx, hit_idx]
            chy = py[ray_idx, hit_idx]
            hx[lo:hi] = chx
            hy[lo:hi] = chy
            dist[lo:hi] = np.sqrt((chx - ox) ** 2 + (chy - oy) ** 2)

        return dist.reshape(N, R), hx.reshape(N, R), hy.reshape(N, R)
