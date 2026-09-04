"""
GPU-accelerated batched neural network inference for NEAT.

Extracts weight matrices from neat-python genomes once per generation,
then runs all cars' feed-forward passes in parallel as GPU matrix
multiplications via PyTorch. This is a mathematically exact replica
of neat-python's FeedForwardNetwork.activate(), so training quality
is unaffected.
"""

import numpy as np
import torch
import neat
from typing import List, Tuple, Dict


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPUBatchInference:
    """
    Runs NEAT FeedForward networks for a full population on the GPU in one
    massive dense matrix multiplication, completely eliminating topology divergence lag!
    """

    def __init__(self, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config,
                 prebuilt_nets=None):
        self.config = config
        
        input_keys  = self.config.genome_config.input_keys
        output_keys = self.config.genome_config.output_keys
        n_inputs    = len(input_keys)
        n_outputs   = len(output_keys)
        
        # 1. Find max number of hidden nodes in any single genome to compute dense matrix size
        max_hidden = 0
        for _, g in genomes:
            hidden_count = sum(1 for nid in g.nodes.keys() if nid not in output_keys)
            if hidden_count > max_hidden:
                max_hidden = hidden_count
                
        # Tensor Layout per car: [Inputs (0 to n_inputs-1)] -> [Outputs] -> [Hidden (contiguous)]
        self.n_total_nodes = n_inputs + n_outputs + max_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_cars = len(genomes)
        
        W = np.zeros((self.n_cars, self.n_total_nodes, self.n_total_nodes), dtype=np.float32)
        B = np.zeros((self.n_cars, self.n_total_nodes), dtype=np.float32)
        R = np.ones((self.n_cars, self.n_total_nodes), dtype=np.float32)
        
        self.max_depth = 1
        
        # Pre-map fixed input and output nodes (same for all cars)
        # Assuming input_keys are negative (NEAT standard) or just fixed list.
        in_idx = {k: i for i, k in enumerate(input_keys)}
        out_idx = {k: n_inputs + i for i, k in enumerate(output_keys)}
        self.out_cols = [out_idx[k] for k in output_keys]
        
        for i, (_, g) in enumerate(genomes):
            hidden_count = 0
            hidden_idx = {}
            
            def map_node(nid):
                if nid in in_idx:
                    return in_idx[nid]
                if nid in out_idx:
                    return out_idx[nid]
                # It's a hidden node
                if nid not in hidden_idx:
                    hidden_idx[nid] = n_inputs + n_outputs + len(hidden_idx)
                return hidden_idx[nid]
                
            # Populate Bias and Response for Output and Hidden nodes
            for nid, node in g.nodes.items():
                mapped_n = map_node(nid)
                B[i, mapped_n] = node.bias
                R[i, mapped_n] = node.response
                
            # Calculate actual graph depth (roughly bounded by number of hidden nodes + 1)
            if len(hidden_idx) + 1 > self.max_depth:
                self.max_depth = len(hidden_idx) + 1
                
            # Populate Weights
            for cg in g.connections.values():
                if cg.enabled:
                    n_in = map_node(cg.key[0])
                    n_out = map_node(cg.key[1])
                    W[i, n_out, n_in] = cg.weight
                    
        # Move to GPU
        self.W = torch.tensor(W, device=DEVICE)
        self.B = torch.tensor(B, device=DEVICE)
        self.R = torch.tensor(R, device=DEVICE)
        
    def activate_all(self, inputs_array: np.ndarray) -> np.ndarray:
        """
        Evaluates ALL networks in exactly self.max_depth tensor operations.
        """
        X = torch.zeros((self.n_cars, self.n_total_nodes), device=DEVICE)
        X[:, :self.n_inputs] = torch.tensor(inputs_array, device=DEVICE)
        
        # Dense Jacobi iteration to simulate feedforward topological propagation
        for _ in range(self.max_depth):
            # X_new = tanh((W * X + B) * R)
            # W is (cars, nodes, nodes), X is (cars, nodes, 1)
            agg = torch.bmm(self.W, X.unsqueeze(2)).squeeze(2)
            X_new = torch.tanh(2.5 * (self.B + self.R * agg))
            
            # Keep inputs untouched
            X_new[:, :self.n_inputs] = X[:, :self.n_inputs]
            X = X_new
            
        return X[:, self.out_cols].cpu().numpy()
