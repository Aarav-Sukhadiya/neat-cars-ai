# ------------------ IMPORTS ------------------


import neat
import pygame
from src.render.neural_network.node import Node, Connection, NodeType
from src.render.colors import Color


# ------------------ CLASSES ------------------


class NN:
    INPUT_NEURONS = 7
    OUTPUT_NEURONS = 4

    def __init__(self, config: neat.Config, genome: neat.DefaultGenome, pos: tuple):
        self.nodes = []
        self.genome = genome
        self.pos = (int(pos[0]+Node.RADIUS), int(pos[1]))
        input_names = ["-90°", "-45°", "-20°", "0°", "20°", "45°", "90°"]
        output_names = ["Left", "Right", "Accelerate", "Brake"]
        hidden_nodes = list(genome.nodes.keys())   # outputs are removed below
        node_id_list = []

        # nodes
        h = (NN.INPUT_NEURONS-1)*(Node.RADIUS*2 + Node.SPACING)
        for i, input_key in enumerate(config.genome_config.input_keys):
            n = Node(input_key, pos[0], pos[1]+int(-h/2 + i*(Node.RADIUS*2 + Node.SPACING)), NodeType.INPUT, [
                     Color.GREEN_PALE, Color.GREEN, Color.DARK_GREEN_PALE, Color.DARK_GREEN], input_names[i], i)
            self.nodes.append(n)
            node_id_list.append(input_key)

        h = (NN.OUTPUT_NEURONS-1)*(Node.RADIUS*2 + Node.SPACING)
        for i, out in enumerate(config.genome_config.output_keys):
            n = Node(out, pos[0] + 2*(Node.LAYER_SPACING+2*Node.RADIUS), pos[1]+int(-h/2 + i*(
                Node.RADIUS*2 + Node.SPACING)), NodeType.OUTPUT, [Color.RED_PALE, Color.RED, Color.DARK_RED_PALE, Color.DARK_RED], output_names[i], i)
            self.nodes.append(n)
            hidden_nodes.remove(out)
            node_id_list.append(out)

        h = (len(hidden_nodes)-1)*(Node.RADIUS*2 + Node.SPACING)
        for i, m in enumerate(hidden_nodes):
            n = Node(m, self.pos[0] + (Node.LAYER_SPACING+2*Node.RADIUS), self.pos[1]+int(-h/2 + i*(Node.RADIUS*2 +
                     Node.SPACING)), NodeType.HIDDEN, [Color.BLUE_PALE, Color.DARK_BLUE, Color.BLUE_PALE, Color.DARK_BLUE])
            self.nodes.append(n)
            node_id_list.append(m)

        # connections
        node_by_id = dict(zip(node_id_list, self.nodes))
        self.connections = []
        for c in genome.connections.values():
            if c.enabled:
                in_node = node_by_id.get(c.key[0])
                out_node = node_by_id.get(c.key[1])
                if in_node is not None and out_node is not None:
                    self.connections.append(Connection(in_node, out_node, c.weight))

    def draw(self, screen: pygame.Surface):
        for c in self.connections:
            c.draw(screen)
        for node in self.nodes:
            node.draw(screen)
