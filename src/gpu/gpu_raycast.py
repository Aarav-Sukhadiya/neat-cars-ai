"""
GPU-accelerated raycast sensor computation for all cars simultaneously.

The track surface is converted once per frame to a boolean NumPy/CuPy
array. All 2000 cars × 5 rays are then marched in parallel on the GPU
using CuPy, replacing the per-car per-ray pixel-walking Python loop
in car.py.

Falls back gracefully to NumPy (CPU) if CUDA/CuPy is unavailable.
"""

import numpy as np
import math

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


# Sensor angles relative to the car's heading (degrees)
SENSOR_ANGLES_DEG = [-90, -45, 0, 45, 90]
MAX_RAY_STEPS = 1920          # matches Car.SENSORS_DRAW_DISTANCE


# ---------------------------------------------------------------------------
# CUDA kernel source (used when CuPy is available)
# ---------------------------------------------------------------------------

_RAYCAST_KERNEL_SOURCE = r"""
extern "C" __global__
void raycast_kernel(
    const unsigned char* track_pixels,  // H x W, 1 = wall (white), 0 = track
    int track_w,
    int track_h,
    const float* cx,          // center_x per car  [N]
    const float* cy,          // center_y per car  [N]
    const float* cos_arr,     // cos(ray_angle)   [N * R]
    const float* sin_arr,     // sin(ray_angle)   [N * R]
    float* out_dist,          // output distances [N * R]
    float* out_hx,            // hit x           [N * R]
    float* out_hy,            // hit y           [N * R]
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
    out_dist[idx] = dist;
    out_hx[idx]   = px;
    out_hy[idx]   = py;
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
        self._cached_track_hash = None
        self._track_gpu = None
        self.use_gpu = _CUPY_AVAILABLE

    def _update_track(self, track_pixels_np: np.ndarray):
        """Upload the track pixel array to the GPU if it changed.
        
        IMPORTANT: The array must be C-contiguous (row-major) because the CUDA
        kernel indexes with [iy * track_w + ix].
        """
        # Zero-cost check: if the array object is exactly the same, we already uploaded it.
        track_id = id(track_pixels_np)
        if getattr(self, '_cached_track_id', None) == track_id:
            return

        self._cached_track_id = track_id
        
        # Force C-contiguous layout
        track_c = np.ascontiguousarray(track_pixels_np)
        if self.use_gpu:
            self._track_gpu = cp.asarray(track_c)
        else:
            self._track_np = track_c

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

        # Build ray direction arrays: (N*R,)
        # angle = 360 - (heading + sensor_offset)  [matches car.py convention]
        sensor_offsets = np.array(SENSOR_ANGLES_DEG, dtype=np.float32)

        # Expand: headings (N,) -> (N,1), offsets (R,) -> (1,R)
        ray_angles_deg = (360.0 - (angles_deg[:, None] + sensor_offsets[None, :]))  # (N,R)
        ray_angles_rad = np.deg2rad(ray_angles_deg).astype(np.float32)              # (N,R)

        cos_vals = np.cos(ray_angles_rad).reshape(-1).astype(np.float32)  # (N*R,)
        sin_vals = np.sin(ray_angles_rad).reshape(-1).astype(np.float32)

        if self.use_gpu:
            return self._compute_gpu(centers_x, centers_y, cos_vals, sin_vals, N, R)
        else:
            return self._compute_cpu(track_pixels_np, centers_x, centers_y, cos_vals, sin_vals, N, R)

    def _compute_gpu(self, cx, cy, cos_vals, sin_vals, N, R):
        kernel = _get_kernel()

        cx_gpu   = cp.asarray(cx.astype(np.float32))
        cy_gpu   = cp.asarray(cy.astype(np.float32))
        cos_gpu  = cp.asarray(cos_vals)
        sin_gpu  = cp.asarray(sin_vals)
        dist_gpu = cp.zeros(N * R, dtype=cp.float32)
        hx_gpu   = cp.zeros(N * R, dtype=cp.float32)
        hy_gpu   = cp.zeros(N * R, dtype=cp.float32)

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
                dist_gpu, hx_gpu, hy_gpu,
                np.int32(MAX_RAY_STEPS),
                np.int32(N),
                np.int32(R),
            )
        )

        distances = cp.asnumpy(dist_gpu).reshape(N, R)
        hit_x     = cp.asnumpy(hx_gpu).reshape(N, R)
        hit_y     = cp.asnumpy(hy_gpu).reshape(N, R)
        return distances, hit_x, hit_y

    def _compute_cpu(self, track_pixels_np, cx, cy, cos_vals, sin_vals, N, R):
        """NumPy CPU fallback — still vectorised (no Python loops per ray)."""
        track_w = self.track_w
        track_h = self.track_h

        steps = np.arange(1, MAX_RAY_STEPS + 1, dtype=np.float32)  # (S,)

        # Expand dims for broadcasting: (N*R, S)
        ox = np.repeat(cx, R).astype(np.float32)  # (N*R,)
        oy = np.repeat(cy, R).astype(np.float32)

        # px/py for every step: (N*R, S)
        px = ox[:, None] + cos_vals[:, None] * steps[None, :]
        py = oy[:, None] + sin_vals[:, None] * steps[None, :]

        ix = np.clip(px.astype(np.int32), 0, track_w - 1)
        iy = np.clip(py.astype(np.int32), 0, track_h - 1)

        # Out-of-bounds mask
        oob = (px < 0) | (px >= track_w) | (py < 0) | (py >= track_h)

        # Wall hit mask
        wall = track_pixels_np[iy, ix] != 0

        hit_mask = wall | oob  # (N*R, S)

        # First hit step index
        hit_idx = np.argmax(hit_mask, axis=1)  # (N*R,)
        # If no hit at all, argmax returns 0; check
        no_hit = ~hit_mask.any(axis=1)
        hit_idx[no_hit] = MAX_RAY_STEPS - 1

        # Gather hit coordinates
        ray_idx = np.arange(N * R)
        hx = px[ray_idx, hit_idx]
        hy = py[ray_idx, hit_idx]

        dist = np.sqrt((hx - ox)**2 + (hy - oy)**2)

        return (
            dist.reshape(N, R).astype(np.float32),
            hx.reshape(N, R).astype(np.float32),
            hy.reshape(N, R).astype(np.float32),
        )
