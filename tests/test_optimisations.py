"""Exactness tests for the performance optimisations (they must not change results)."""

import numpy as np
import neat

from src.gpu.gpu_inference import GPUBatchInference, _longest_path
from src.gpu.gpu_raycast import GPURaycast
from test_gpu import _make_neat_config


def _mutated_genomes(config, n=30, rounds=8, seed=3):
    import random
    random.seed(seed)
    pop = neat.Population(config)
    genomes = list(pop.population.items())[:n]
    config.genome_config.node_add_prob = 0.6
    config.genome_config.conn_add_prob = 0.6
    for _, g in genomes:
        for _ in range(rounds):
            g.mutate(config.genome_config)
    return genomes


def test_longest_path_examples():
    inputs = [-1, -2]
    # -1 -> 5 -> 6 -> 0, and -2 -> 0 : longest path 3 edges
    assert _longest_path(inputs, [0, 5, 6], [(-1, 5), (5, 6), (6, 0), (-2, 0)]) == 3
    # isolated output with no inputs is a "bias neuron": settles after 1 iteration
    assert _longest_path(inputs, [0], []) == 1


def test_gpu_inference_matches_neat_python_on_deep_networks():
    config = _make_neat_config()
    genomes = _mutated_genomes(config)
    gpu = GPUBatchInference(genomes, config)
    rng = np.random.default_rng(0)
    inputs = rng.uniform(0, 300, size=(len(genomes), len(config.genome_config.input_keys))).astype(np.float32)
    out = gpu.activate_all(inputs)
    for i, (_, g) in enumerate(genomes):
        ref = neat.nn.FeedForwardNetwork.create(g, config).activate(inputs[i].tolist())
        np.testing.assert_allclose(out[i], ref, atol=1e-4)


def test_gpu_inference_row_subset_is_identical():
    config = _make_neat_config()
    genomes = _mutated_genomes(config)
    gpu = GPUBatchInference(genomes, config)
    assert len(genomes) >= 8
    inputs = np.random.default_rng(1).uniform(0, 300, (len(genomes), 7)).astype(np.float32)
    rows = np.array([0, 3, 4, len(genomes) - 1])
    np.testing.assert_array_equal(gpu.activate_all(inputs, rows=rows), gpu.activate_all(inputs)[rows])


def _box_track(w=400, h=300):
    t = np.zeros((h, w), dtype=np.uint8)
    t[:10, :] = t[-10:, :] = t[:, :10] = t[:, -10:] = 255
    t[100:140, 200:240] = 255
    return t


def test_cpu_raycast_chunking_is_exact(monkeypatch):
    track = _box_track()
    rc = GPURaycast(400, 300)
    rc.use_gpu = False
    rng = np.random.default_rng(2)
    cx, cy = rng.uniform(20, 380, 300).astype(np.float32), rng.uniform(20, 280, 300).astype(np.float32)
    ang = rng.uniform(0, 360, 300).astype(np.float32)
    small = rc.compute(track, cx, cy, ang)
    monkeypatch.setattr(GPURaycast, "CPU_CHUNK_RAYS", 10 ** 9)   # single chunk = old behaviour
    big = rc.compute(track, cx, cy, ang)
    for a, b in zip(small, big):
        np.testing.assert_array_equal(a, b)


def test_raycast_empty_batch():
    rc = GPURaycast(400, 300)
    d, hx, hy = rc.compute(_box_track(), np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
    assert d.shape == hx.shape == hy.shape == (0, 7)


def test_raycast_track_cache_follows_new_arrays():
    """The cache used id(); a new array reusing a freed id would have kept the stale track."""
    rc = GPURaycast(400, 300)
    c = np.array([200.0], np.float32)
    y = np.array([60.0], np.float32)
    down = np.array([270.0], np.float32)   # heading 270 = +y (down the screen)
    d1, _, _ = rc.compute(_box_track(), c, y, down)       # hits the block at y=100
    open_track = _box_track()
    open_track[100:140, 200:240] = 0
    d2, _, _ = rc.compute(open_track, c, y, down)         # now reaches the bottom wall
    assert d2[0, 3] > d1[0, 3] + 50
