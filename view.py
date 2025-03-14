""" Display WSN """

import matplotlib.pyplot as plt

def display_nodes(nodes):
    """Displays a set of nodes on a 2D plot."""
    plt.figure(figsize=(10, 10))
    for node in nodes:
        if node.is_sink:
            plt.scatter(node.x, node.y, color=node.color, s=100, marker="^", edgecolors="black")  # Triangle for sink node
        else:
            plt.scatter(node.x, node.y, color=node.color, s=50, marker="o", edgecolors="black")  # Circle for normal nodes
    
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Sensor Network Nodes")
    plt.grid(True)
    plt.show()
