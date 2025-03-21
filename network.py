"""Network Setup"""
W = 1000
H = 1000
from entity import Node 
import matplotlib.pyplot as plt
import random
import json
from datetime import datetime
from buffer import ClusterHeadBuffer
class Network:
    def __init__(self, num_nodes, width = W, height = H):
        self.width = width 
        self.height = height
        self.num_nodes = num_nodes
        self.num_nodes_history = []
        self.nodes = self._generate_nodes()
        self.available_nodes = self.nodes
        self.sink_node = self._select_sink()
        self.cluster_heads = []
        self.cluster_heads_2 = []
        self.cluster_infos = []
        self.cluster_2_infos = []
        self.edges = []
        self.edge_colors = {}  # Dictionary to store edge colors
        self.time_life = 0 
        self.cluster_heads_buffer = ClusterHeadBuffer(mem_size=20)
        self.cluster_heads_2_buffer = ClusterHeadBuffer(mem_size=20)
        self.error = "None"
    
    def _generate_nodes(self):
        """Sinh ngẫu nhiên các node trong miền (0,0) đến (width, height)"""
        return [Node(random.uniform(0, self.width), random.uniform(0, self.height)) for _ in range(self.num_nodes)]
    
    def _select_sink(self):
        """Chọn ngẫu nhiên một node làm sink"""
        sink = random.choice(self.nodes)
        sink.is_sink = True
        sink.update_status()
        return sink
    
    def add_edge(self, source_id, target_id, color='black'):
        """Thêm các cạnh dựa trên id"""
        self.edges.append((source_id, target_id))
        self.edge_colors[(source_id, target_id)] = color
    
    def reset(self):
        """ Reset """
        self.cluster_heads = []
        self.cluster_heads_2 = []
        self.cluster_infos = []
        self.cluster_2_infos = []
        self.edges = []
        self.edge_colors = {}  # Dictionary to store edge colors
        self.error = "None"

    
    def display_nodes(self):
        """In ra danh sách các node"""
        for i, node in enumerate(self.available_nodes):
            sink_status = " (Sink Node)" if getattr(node, 'is_sink', False) else ""
            print(f"Node {i}: (x={node.x:.2f}, y={node.y:.2f}){sink_status}")
    
    def display_edges(self):
        """In ra danh sách các cạnh trong mạng"""
        print("Edges:")
        for source, target in self.edges:
            print(f"(Node {self.available_nodes.index(source)}, Node {self.available_nodes.index(target)})")
    
    def take_top(self, M):
        """Chọn M nodes có energy cao nhất từ available_nodes, không chọn sink."""
        return sorted(
            [node for node in self.available_nodes if not node.is_sink], 
            key=lambda node: node.energy, 
            reverse=True
        )[:M]


    def display_network(self):
        """ Hiển thị toàn bộ WSN """
        plt.figure(figsize=(10, 10))
        for node in self.nodes:
            if node.is_sink:
                plt.scatter(node.x, node.y, color=node.color, s=150, marker="^", edgecolors="black")  # Triangle for sink node
            else:
                plt.scatter(node.x, node.y, color=node.color, s=50, marker="o", edgecolors="black")  # Circle for normal nodes
        for edge in self.edges:
            source_id, target_id = edge
            source_node = Node.nodes[source_id]
            target_node = Node.nodes[target_id]
            color = self.edge_colors.get((source_id, target_id), 'black')  # Get color or default to black
            plt.plot([source_node.x, target_node.x], [source_node.y, target_node.y], color=color, alpha=1.0)  # Draw edges
        
         # Hiển thị số vòng đời hiện tại
        plt.text(0.05, 1.01, f"Life time: {self.time_life}", transform=plt.gca().transAxes, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
        plt.text(0.05, 1.05, f"Available nodes: {len(self.available_nodes)}", transform=plt.gca().transAxes, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
        plt.text(0.4, 1.05, f"Error: {self.error}", transform=plt.gca().transAxes, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Wireless Sensor Network Visualization")
        plt.grid(True)
        plt.show()
    def update_network(self):
        """Cập nhật mạng: loại bỏ các node chết và giảm số lượng node"""
        self.available_nodes = [node for node in self.available_nodes if not node.is_dead]  # Giữ lại node còn sống
        self.num_nodes = len(self.available_nodes)  # Cập nhật số lượng node
        self.num_nodes_history.append(self.num_nodes)
        print(f"Network updated: {self.num_nodes} nodes remaining.")

    def save_network(self, filename="network"):
        """Lưu toàn bộ thông tin của mạng vào một file JSON với timestamp trong tên file"""

        data = {
            "width": self.width,
            "height": self.height,
            "num_nodes_history": self.num_nodes_history,
            "sink": {
                "x": self.sink_node.x,
                "y": self.sink_node.y
            },
            "nodes": [
                {
                    "id": node.id,
                    "x": node.x,
                    "y": node.y,
                    "is_cluster_head": node.is_cluster_head_history,
                    "energy": node.energy_history,
                    "is_dead": node.life_history,
                    "cluster_index": node.cluster_index_history
                }
                for node in self.nodes
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Network saved to {filename}")
