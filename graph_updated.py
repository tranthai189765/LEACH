from itertools import combinations
from collections import deque

class Graph:
    def __init__(self, nodes, R):
        self.nodes = nodes
        self.R = R
        self.adj_list = {node.id: [] for node in nodes}
        self.edges = []
        self.create_graph()
    
    def create_graph(self):
        """Tạo đồ thị dựa trên bán kính R."""
        for node1, node2 in combinations(self.nodes, 2):
            if node1.distance_to(node2) <= self.R:
                self.add_edge(node1.id, node2.id, node1.distance_to(node2))
    
    def add_edge(self, node1_id, node2_id, weight):
        """Thêm cạnh vào đồ thị với trọng số."""
        self.adj_list[node1_id].append(node2_id)
        self.adj_list[node2_id].append(node1_id)
        self.edges.append((node1_id, node2_id, weight))

    def find_mst(self):
        """Tìm cây khung nhỏ nhất (MST) bằng Kruskal."""
        parent = {node.id: node.id for node in self.nodes}
        
        def find(node_id):
            if parent[node_id] != node_id:
                parent[node_id] = find(parent[node_id])
            return parent[node_id]

        def union(node1_id, node2_id):
            root1 = find(node1_id)
            root2 = find(node2_id)
            if root1 != root2:
                parent[root2] = root1

        sorted_edges = sorted(self.edges, key=lambda edge: edge[2])
        mst_edges = []
        for node1_id, node2_id, weight in sorted_edges:
            if find(node1_id) != find(node2_id):
                union(node1_id, node2_id)
                mst_edges.append((node1_id, node2_id))
        return mst_edges

    def is_connected_with_component(self):
        """Kiểm tra liên thông, và trả về thành phần liên thông chứa node.isSink == True nếu không liên thông."""
        def bfs(start_id):
            """Tìm một thành phần liên thông bắt đầu từ node `start_id`."""
            visited = set()
            queue = deque([start_id])
            component = []

            while queue:
                node_id = queue.popleft()
                if node_id not in visited:
                    visited.add(node_id)
                    component.append(node_id)
                    queue.extend(neigh for neigh in self.adj_list[node_id] if neigh not in visited)
            return component

        visited_global = set()
        components = []

        for node in self.nodes:
            if node.id not in visited_global:
                component = bfs(node.id)
                visited_global.update(component)
                components.append(component)

        if len(components) == 1:
            return True, components[0]  # Đồ thị liên thông

        # Nếu không liên thông, tìm component chứa node có isSink == True
        sink_component = []
        for component in components:
            for node in self.nodes:
                if node.id in component and getattr(node, 'is_sink', False):
                    sink_component = component
                    break
            if sink_component:
                break

        return False, sink_component
