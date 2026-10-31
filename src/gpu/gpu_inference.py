"""
GPU-accelerated batched neural network inference for NEAT.

Every genome of the generation is packed into one dense weight tensor
``W[car, to_node, from_node]``, then all networks are evaluated together with
batched matrix multiplies (PyTorch, CUDA if available, else CPU).

Equivalence with ``neat.nn.FeedForwardNetwork.activate``
--------------------------------------------------------
Each iteration computes, for every non-input node,
``x = tanh(2.5 * (bias + response * sum(w * x_in)))`` — exactly neat-python's
``tanh_activation(bias + response * sum_aggregation(...))``. Repeating this
"Jacobi" update ``depth`` times, where ``depth`` is the length of the longest
path in the network, yields the exact feed-forward values (nodes with no
inputs — neat-python's "bias neurons" — settle after the first iteration).
Hidden nodes that neat-python prunes (not required for any output) are still
computed here, but by definition they cannot influence an output.

Assumes ``activation = tanh`` and ``aggregation = sum`` for every node, which
is what ``config/neat_config.ini`` configures.
"""

from typing import List, Tuple

import neat
import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _longest_path(input_keys, node_keys, edges) -> int:
    """Number of Jacobi iterations needed: longest path (in edges) from any source.

    Inputs have level 0; any other node has level 1 + max(level of predecessors),
    or 1 if it has none. Returns the maximum level (>= 1).
    """
    preds = {n: [] for n in node_keys}
    for a, b in edges:
        preds.setdefault(b, []).append(a)
    inputs = set(input_keys)
    level = {}

    def lvl(n):
        if n in inputs:
            return 0
        if n in level:
            return level[n]
        # Iterative DFS to avoid recursion limits on very deep genomes.
        stack = [n]
        while stack:
            top = stack[-1]
            pending = [p for p in preds.get(top, ()) if p not in inputs and p not in level]
            if pending:
                stack.extend(pending)
                continue
            stack.pop()
            ps = preds.get(top, ())
            level[top] = 1 + max((0 if p in inputs else level[p] for p in ps), default=0)
        return level[n]

    return max((lvl(n) for n in preds), default=1)


class GPUBatchInference:
    """Runs a whole population's NEAT feed-forward networks in a few batched GPU ops."""

    def __init__(self, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config,
                 prebuilt_nets=None):
        # ``prebuilt_nets`` is accepted for API compatibility; the tensors are built
        # straight from the genomes so no FeedForwardNetwork is needed.
        self.config = config

        input_keys = config.genome_config.input_keys
        output_keys = config.genome_config.output_keys
        n_inputs = len(input_keys)
        n_outputs = len(output_keys)
        output_set = set(output_keys)

        max_hidden = 0
        for _, g in genomes:
            hidden_count = sum(1 for nid in g.nodes if nid not in output_set)
            max_hidden = max(max_hidden, hidden_count)

        # Node layout per car: [inputs][outputs][hidden...]
        self.n_total_nodes = n_inputs + n_outputs + max_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_cars = len(genomes)

        W = np.zeros((self.n_cars, self.n_total_nodes, self.n_total_nodes), dtype=np.float32)
        B = np.zeros((self.n_cars, self.n_total_nodes), dtype=np.float32)
        R = np.ones((self.n_cars, self.n_total_nodes), dtype=np.float32)

        in_idx = {k: i for i, k in enumerate(input_keys)}
        out_idx = {k: n_inputs + i for i, k in enumerate(output_keys)}
        self.out_cols = [out_idx[k] for k in output_keys]

        self.max_depth = 1
        for i, (_, g) in enumerate(genomes):
            hidden_idx = {}

            def map_node(nid):
                if nid in in_idx:
                    return in_idx[nid]
                if nid in out_idx:
                    return out_idx[nid]
                if nid not in hidden_idx:
                    hidden_idx[nid] = n_inputs + n_outputs + len(hidden_idx)
                return hidden_idx[nid]

            for nid, node in g.nodes.items():
                mapped_n = map_node(nid)
                B[i, mapped_n] = node.bias
                R[i, mapped_n] = node.response

            edges = []
            for cg in g.connections.values():
                if cg.enabled:
                    W[i, map_node(cg.key[1]), map_node(cg.key[0])] = cg.weight
                    edges.append(cg.key)

            self.max_depth = max(self.max_depth, _longest_path(input_keys, g.nodes.keys(), edges))

        self.W = torch.from_numpy(W).to(DEVICE)
        self.B = torch.from_numpy(B).to(DEVICE)
        self.R = torch.from_numpy(R).to(DEVICE)
        self._out_cols_t = torch.tensor(self.out_cols, device=DEVICE)

    def activate_all(self, inputs_array: np.ndarray, rows: np.ndarray = None) -> np.ndarray:
        """Evaluate the networks.

        Args:
            inputs_array: (n_cars, n_inputs) float32 sensor inputs for every car.
            rows: optional index array; if given, only those cars are evaluated and the
                  result has ``len(rows)`` rows (in the same order).

        Returns:
            (n_rows, n_outputs) float32 network outputs.
        """
        if rows is None:
            W, B, R = self.W, self.B, self.R
            inputs = inputs_array
        else:
            rows_t = torch.from_numpy(np.asarray(rows, dtype=np.int64)).to(DEVICE)
            W, B, R = self.W[rows_t], self.B[rows_t], self.R[rows_t]
            inputs = inputs_array[rows]

        X = torch.zeros((W.shape[0], self.n_total_nodes), device=DEVICE)
        X_in = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)).to(DEVICE)
        X[:, :self.n_inputs] = X_in

        for _ in range(self.max_depth):
            agg = torch.bmm(W, X.unsqueeze(2)).squeeze(2)
            X = torch.tanh(2.5 * (B + R * agg))
            X[:, :self.n_inputs] = X_in   # inputs are clamped

        return X[:, self._out_cols_t].cpu().numpy()
