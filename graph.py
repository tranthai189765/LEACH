import heapq
from itertools import combinations

class Graph:
    """Graph representation using adjacency list."""
    def __init__(self, nodes):
        self.nodes = nodes
        self.adj_list = {node.id: [] for node in nodes}
        self.create_full_graph()
    
    def create_full_graph(self):
        """Ensure the graph is fully connected by adding all possible edges."""
        for node1, node2 in combinations(self.nodes, 2):
            weight = self.calculate_weight(node1, node2)
            self.add_edge(node1.id, node2.id, weight)
    
    def calculate_weight(self, node1, node2):
        """Calculate weight between two nodes. Can be modified as needed."""
        return ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5  # Euclidean distance

    def add_edge(self, node1_id, node2_id, weight):
        """ Add edge """
        self.adj_list[node1_id].append((node2_id, weight))
        self.adj_list[node2_id].append((node1_id, weight))

    def prim_mst(self):
        """Implements Prim's algorithm to find the Minimum Spanning Tree (MST)."""
        if not self.nodes:
            return None, 0
        
        mst = []  # List to store edges of MST
        total_weight = 0
        visited = set()
        min_heap = [(0, self.nodes[0].id, -1)]  # (weight, node_id, parent_id)
        
        while min_heap and len(visited) < len(self.nodes):
            weight, node_id, parent_id = heapq.heappop(min_heap)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            if parent_id != -1:
                mst.append((parent_id, node_id))  # Store only (node1, node2)
                total_weight += weight
            
            for neighbor_id, edge_weight in self.adj_list[node_id]:
                if neighbor_id not in visited:
                    heapq.heappush(min_heap, (edge_weight, neighbor_id, node_id))
        
        return mst, total_weight

