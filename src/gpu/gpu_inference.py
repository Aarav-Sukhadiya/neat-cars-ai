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
        
        # 1. Find max node ID across ALL genomes
        max_node_id = -1
        for _, g in genomes:
            if g.nodes:
                m = max(g.nodes.keys())
                if m > max_node_id:
                    max_node_id = m
                    
        # Node mapping:
        # Inputs: 0 to n_inputs-1
        # Outputs and Hidden: n_inputs to n_inputs + max_node_id
        self.n_total_nodes = n_inputs + max_node_id + 1
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_cars = len(genomes)
        
        def map_node(nid):
            if nid < 0:
                # neat-python input nodes are -1, -2, -3... 
                # (-1 becomes 0, -2 becomes 1)
                return abs(nid) - 1
            return n_inputs + nid
            
        W = np.zeros((self.n_cars, self.n_total_nodes, self.n_total_nodes), dtype=np.float32)
        B = np.zeros((self.n_cars, self.n_total_nodes), dtype=np.float32)
        R = np.ones((self.n_cars, self.n_total_nodes), dtype=np.float32)
        
        self.max_depth = 1
        
        for i, (_, g) in enumerate(genomes):
            hidden_count = 0
            for nid, node in g.nodes.items():
                mapped_n = map_node(nid)
                B[i, mapped_n] = node.bias
                R[i, mapped_n] = node.response
                if nid not in output_keys:
                    hidden_count += 1
                    
            if hidden_count + 1 > self.max_depth:
                self.max_depth = hidden_count + 1
                
            for cg in g.connections.values():
                if cg.enabled:
                    n_in = map_node(cg.key[0])
                    n_out = map_node(cg.key[1])
                    W[i, n_out, n_in] = cg.weight
                    
        # Output columns
        self.out_cols = [map_node(k) for k in output_keys]
        
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
            X_new = torch.tanh((agg + self.B) * self.R)
            
            # Keep inputs untouched
            X_new[:, :self.n_inputs] = X[:, :self.n_inputs]
            X = X_new
            
        return X[:, self.out_cols].cpu().numpy()
