""" Entities """
import random

ENEGY_0 = 1000  # Initial energy
W = 100
H = 100
class Node:
    """
    Represents a node in a sensor network.

    Attributes:
        id (int): Unique identifier of the node.
        energy (float): The current energy level of the node.
        is_dead (bool): Indicates whether the node is dead.
        is_cluster_head (bool): Indicates whether the node is a cluster head.
        x (float): The x-coordinate of the node.
        y (float): The y-coordinate of the node.
    """
    id_counter = 0
    nodes = {}  # Dictionary to store nodes by ID

    def __init__(self, x=None, y=None):
        self.id = Node.id_counter
        Node.id_counter += 1
        self.energy = ENEGY_0
        self.is_dead = False
        self.is_cluster_head = False
        self.is_cluster_head_2 = False
        self.is_sink = False
        self.cluster_index = 0
        self.cluster_2_index = 0
        self.x = x if x is not None else random.uniform(0, W)
        self.y = y if y is not None else random.uniform(0, H)
        self.color = "green"
        Node.nodes[self.id] = self  # Store node in dictionary
        self.energy_history = []
        self.life_history = []
        self.is_cluster_head_history = []
        self.is_cluster_head_2_history = []
        self.cluster_index_history = []
        self.cluster_2_index_history = []
        self.probability = 0
        self.degree = 1
    
    def update_status(self):
        """Updates the node's color based on its state."""
        if self.energy < 0:
            self.is_dead = True 
        
        if self.is_dead:
            self.color = "black"
        elif self.is_cluster_head_2:
            self.color = "yellow"
        elif self.is_cluster_head:
            self.color = "red"
        elif self.is_sink:
            self.color = "blue"
        elif self.energy < 200:
            self.color = "green"
        else:
            self.color = "green"
    
    def update_all(self):
        self.update_status()
        self.energy_history.append(self.energy)
        self.life_history.append(self.is_dead)
        self.is_cluster_head_history.append(self.is_cluster_head)
        self.cluster_index_history.append(self.cluster_index)
        self.is_cluster_head_2_history.append(self.cluster_2_index)
        self.cluster_2_index_history.append(self.cluster_2_index)

    def distance_to(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)  # Khoảng cách Manhattan
    
    def distance_energy_to(self, other):
        return self.distance_to(other)/other.energy  # Khoảng cách Manhattan

    def __repr__(self):
        return f"Node(x={self.x:.2f}, y={self.y:.2f}, energy={self.energy}, is_dead={self.is_dead}, is_cluster_head={self.is_cluster_head})"

