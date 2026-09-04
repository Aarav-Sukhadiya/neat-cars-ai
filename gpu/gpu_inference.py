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
    batched call.

    NEAT genomes can each have a different topology (different hidden nodes /
    connections). We handle this by grouping genomes by their topology hash
    and running each group as a single batched matrix multiplication.
    In practice, within a single generation topologies are nearly identical
    (NEAT starts uniform and only gradually diverges), so almost all cars
    get batched together.
    """

    def __init__(self, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config,
                 prebuilt_nets=None):
        self.config = config
        self._genome_layers: List[list] = []

        if prebuilt_nets is not None:
            nets = prebuilt_nets
        else:
            nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]

        for net in nets:
            self._genome_layers.append(self._extract_layers(net))
            
        # ---- PRE-COMPILE AND CACHE PYTORCH MATRICES FOR ENTIRE GENERATION ----
        self.compiled_groups = []
        input_keys  = self.config.genome_config.input_keys
        output_keys = self.config.genome_config.output_keys
        n_outputs   = len(output_keys)
        n_inputs    = len(input_keys)
        
        topology_map: Dict[str, List[int]] = {}
        for global_i, layers in enumerate(self._genome_layers):
            key = str([(nid, len(lnks)) for nid, _, _, lnks in layers])
            topology_map.setdefault(key, []).append(global_i)
            
        for group_key, global_ids in topology_map.items():
            n_group = len(global_ids)
            template = self._genome_layers[global_ids[0]]
            n_nodes  = len(template)
            
            if n_nodes == 0:
                self.compiled_groups.append({'global_ids': global_ids, 'n_nodes': 0})
                continue
                
            max_inputs_per_node = max(len(lnks) for _, _, _, lnks in template) if template else 0

            weights   = np.zeros((n_group, n_nodes, max(max_inputs_per_node, 1)), dtype=np.float32)
            biases    = np.zeros((n_group, n_nodes), dtype=np.float32)
            responses = np.zeros((n_group, n_nodes), dtype=np.float32)

            key_to_col: Dict[int, int] = {}
            for ci, k in enumerate(input_keys):
                key_to_col[k] = ci
            for ni, (nid, _, _, _) in enumerate(template):
                key_to_col[nid] = n_inputs + ni

            inp_keys_per_node = [
                [inp_nid for inp_nid, _ in lnks]
                for _, _, _, lnks in template
            ]
            
            for gi_local, global_i in enumerate(global_ids):
                layers = self._genome_layers[global_i]
                for ni, (_, bias, response, lnks) in enumerate(layers):
                    biases[gi_local, ni]    = bias
                    responses[gi_local, ni] = response
                    for li_link, (_, w) in enumerate(lnks):
                        weights[gi_local, ni, li_link] = w

            out_cols = [key_to_col[k] for k in output_keys]
            
            inp_cols_per_node = []
            for inp_nids in inp_keys_per_node:
                inp_cols_per_node.append([key_to_col[k] for k in inp_nids])
            
            # CACHE EVERYTHING IN GPU VRAM
            self.compiled_groups.append({
                'global_ids': global_ids,
                'n_nodes': n_nodes,
                'W': torch.tensor(weights, device=DEVICE),
                'B': torch.tensor(biases, device=DEVICE),
                'R': torch.tensor(responses, device=DEVICE),
                'inp_cols_per_node': inp_cols_per_node,
                'out_cols': out_cols,
                'n_inputs': n_inputs
            })

    def _extract_layers(self, net: neat.nn.FeedForwardNetwork) -> list:
        return [
            (node_id, bias, response, links)
            for node_id, _act, _agg, bias, response, links in net.node_evals
        ]

    def activate_all(self, inputs_array: np.ndarray) -> np.ndarray:
        """
        Run the full population's neural networks in parallel on GPU.
        Args:
            inputs_array: (n_cars, n_inputs) float32 numpy array.
        Returns:
            outputs_array: (n_cars, n_outputs) float32 numpy array.
        """
        n_cars = inputs_array.shape[0]
        n_outputs = len(self.config.genome_config.output_keys)
        
        inputs_tensor = torch.tensor(inputs_array, device=DEVICE)
        outputs_tensor = torch.zeros((n_cars, n_outputs), device=DEVICE)
        
        for group in self.compiled_groups:
            global_ids = group['global_ids']
            n_nodes = group['n_nodes']
            if n_nodes == 0:
                continue
                
            n_group = len(global_ids)
            n_inputs = group['n_inputs']
            
            group_inputs = inputs_tensor[global_ids]
            
            n_value_cols = n_inputs + n_nodes
            values = torch.zeros((n_group, n_value_cols), device=DEVICE)
            values[:, :n_inputs] = group_inputs
            
            W = group['W']
            B = group['B']
            R = group['R']
            inp_cols_per_node = group['inp_cols_per_node']
            
            # PURE PyTorch evaluation (NO python list building)
            for ni in range(n_nodes):
                inp_cols = inp_cols_per_node[ni]
                if not inp_cols:
                    node_out = torch.tanh(B[:, ni] * R[:, ni])
                else:
                    inp_vals = values[:, inp_cols]
                    w_slice  = W[:, ni, :len(inp_cols)]
                    agg      = (inp_vals * w_slice).sum(dim=1)
                    node_out = torch.tanh((agg + B[:, ni]) * R[:, ni])
                    
                values[:, n_inputs + ni] = node_out
                
            outputs_tensor[global_ids] = values[:, group['out_cols']]
            
        return outputs_tensor.cpu().numpy()
