import os
import neat
import graphviz
import pickle

def create_network_visualization(genome, config):
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Neural Network')
    dot.attr(rankdir='LR', size='12,8', dpi='300')  # Increased size and DPI

    # Define node colors
    input_color = '#E6F3FF'     # Light Blue
    hidden_color = '#F0FFF0'    # Pale Green
    output_color = '#FFF0F0'    # Light Salmon

    # Add input nodes
    with dot.subgraph(name='cluster_input') as c:
        c.attr(color='lightblue', style='filled')
        c.node_attr.update(style='filled', color=input_color)
        for i in range(8):
            c.node(f'in{i}', f'In {i}')
        c.attr(label='Inputs', labeljust='l')

    # Organize hidden nodes into layers
    hidden_nodes = [k for k in genome.nodes.keys() if k not in config.genome_config.output_keys and k not in config.genome_config.input_keys]
    layers = {}
    for node in hidden_nodes:
        layer = 1
        for conn in genome.connections.values():
            if conn.enabled and conn.key[1] == node:
                layer = max(layer, layers.get(conn.key[0], 0) + 1)
        layers[node] = layer

    # Add hidden nodes
    max_layer = max(layers.values()) if layers else 0
    for layer in range(1, max_layer + 1):
        with dot.subgraph(name=f'cluster_hidden_{layer}') as c:
            c.attr(color='lightgreen', style='filled')
            c.node_attr.update(style='filled', color=hidden_color)
            for n in [node for node, l in layers.items() if l == layer]:
                activation = genome.nodes[n].activation
                c.node(str(n), f'Node {n}\n({activation})')
            c.attr(label=f'Hidden Layer {layer}', labeljust='l')

    # Add output nodes
    with dot.subgraph(name='cluster_output') as c:
        c.attr(color='lightsalmon', style='filled')
        c.node_attr.update(style='filled', color=output_color)
        for i in config.genome_config.output_keys:
            activation = genome.nodes[i].activation
            c.node(f'out{i}', f'Out {i}\n({activation})')
        c.attr(label='Outputs', labeljust='l')

    # Add connections
    for cg in genome.connections.values():
        if cg.enabled:
            input, output = cg.key
            input_str = f'in{input}' if input < 0 else f'out{input}' if input in config.genome_config.output_keys else str(input)
            output_str = f'out{output}' if output in config.genome_config.output_keys else str(output)
            color = '#FF000080' if cg.weight < 0 else '#0000FF80'  # Semi-transparent colors
            width = str(0.1 + abs(cg.weight) / 2)
            dot.edge(input_str, output_str, color=color, penwidth=width)

    # Render the graph
    dot.render('neural_network', format='png', cleanup=True)
    print("Neural network visualization has been saved as 'neural_network.png'")


def visualize_winner():
    # Load the configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the winner genome
    with open('winner_genome.pkl', 'rb') as f:
        genome = pickle.load(f)

    # Create the network visualization
    create_network_visualization(genome, config)

if __name__ == "__main__":
    visualize_winner()