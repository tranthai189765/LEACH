import numpy as np
import skfuzzy as fuzz
from entity import Node 
from network import Network 
import random
import math
from chs_buffer import Buffer
import copy
from graph_updated import Graph

# Initialize buffer
buffer = Buffer(5)

def euclidean_distances_np(a, b):
    return np.sqrt(((a[:, np.newaxis, :] - b[np.newaxis, :, :]) ** 2).sum(axis=2))

class FuzzyCMeansClustering:
    def __init__(self, network, num_clusters, R1=300, m=2, max_iter=1000, tol=1e-5):
        self.network = network
        self.num_clusters = num_clusters
        self.m = m  # Fuzzy parameter
        self.max_iter = max_iter  # Maximum iterations
        self.tol = tol  # Convergence tolerance
        self.R1 = R1  # Maximum allowed distance from cluster center
        self.cluster_centers = None
        self.membership_matrix = None
        self.labels = None

    def fit(self):
        """Perform custom Fuzzy C-Means clustering with R1 constraint."""
        nodes = np.array([[node.x, node.y] for node in self.network.available_nodes if not node.is_sink])
        if nodes.size == 0:
            raise ValueError("No nodes available for clustering.")
        data = nodes.T  # Shape (2, N)

        # Initialize cluster centers randomly from data points
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(data.shape[1], self.num_clusters, replace=False)
        self.cluster_centers = data[:, indices].T  # Shape (C, 2)

        for iteration in range(self.max_iter):
            # Compute distances from each point to cluster centers (C, N)
            distances = euclidean_distances_np(self.cluster_centers, data.T)

            # Compute membership matrix with R1 constraint
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = distances[:, np.newaxis, :] / (distances[np.newaxis, :, :] + 1e-8)
            ratios = ratios ** (2 / (self.m - 1))
            denominators = np.sum(ratios, axis=1)
            u = 1 / (denominators + 1e-8)

            # Apply R1 constraint: set membership to zero where distance > R1
            mask = distances > self.R1
            u[mask] = 0

            # Handle points with all memberships zero (assign to nearest cluster)
            column_sums = u.sum(axis=0)
            all_zero_mask = column_sums == 0
            if np.any(all_zero_mask):
                nearest_clusters = np.argmin(distances[:, all_zero_mask], axis=0)
                u[nearest_clusters, np.where(all_zero_mask)[0]] = 1
                column_sums = u.sum(axis=0)

            # Normalize memberships
            u = u / column_sums[np.newaxis, :]

            # Update cluster centers
            u_m = u ** self.m
            new_centers = np.dot(u_m, data.T) / (np.sum(u_m, axis=1, keepdims=True) + 1e-8)

            # Check convergence
            if np.linalg.norm(new_centers - self.cluster_centers) < self.tol:
                break
            self.cluster_centers = new_centers

        # Assign labels based on highest membership
        self.labels = np.argmax(u, axis=0)
        self.membership_matrix = u
        # print("donedone")

        # Select cluster heads considering R1 constraint
        non_accept, temp_cluster_heads, current_cluster_heads_ids = self.select_cluster_heads(self.labels, self.cluster_centers)
        return non_accept, temp_cluster_heads, current_cluster_heads_ids
    def select_cluster_heads(self, cluster_labels, cluster_centers):

        """Chọn Cluster Head theo tiêu chí 65% năng lượng, 35% khoảng cách đến tâm cụm"""
        temp_cluster_heads = []  # Reset cluster_heads

        for i in range(self.num_clusters):
            cluster_nodes = [
                node.id for node, label in zip(self.network.available_nodes, cluster_labels) 
                if label == i and not node.is_sink  # Lọc bỏ node có is_sink = True
            ]

            if not cluster_nodes:
                continue  # Nếu cụm không có node nào thì bỏ qua

            # Lấy tâm cụm hiện tại
            cluster_center = cluster_centers[i]

            # Tính điểm tổng hợp dựa trên năng lượng (65%) và khoảng cách đến tâm cụm (35%)
            def score(node_id):
                node = Node.nodes[node_id]
                energy_score = node.energy / 1000  # Chuẩn hóa năng lượng về [0,1]
                distance = np.linalg.norm(np.array([node.x, node.y]) - cluster_center)  # Khoảng cách đến tâm cụm
                distance_score = 1 / (1 + distance)  # Chuẩn hóa khoảng cách về [0,1], càng gần tâm điểm càng cao
                return energy_score
            # Chọn node có điểm cao nhất làm Cluster Head
            current_cluster_heads_ids = []
            ch_id = max(cluster_nodes, key=score)
            ch_node = Node.nodes[ch_id]
            
            # Lưu vào danh sách Cluster Heads
            temp_cluster_heads.append(ch_node)
            current_cluster_heads_ids.append(ch_node.id)
        
        non_accept, temp_cluster_heads, current_cluster_heads_ids = self.select_more_cluster_heads(temp_cluster_heads, current_cluster_heads_ids)
        # print("donedone")
        return non_accept, temp_cluster_heads, current_cluster_heads_ids


    def select_more_cluster_heads(self, temp_cluster_heads, current_cluster_heads_ids):
        non_accept = 0
        while not self.network.is_coveraged(temp_cluster_heads)[1]:
            # print("this")
            coveraged_set, _ = self.network.is_coveraged(temp_cluster_heads)
            candidate_node_ids_clusters = [node.id for node in self.network.available_nodes if not node.is_sink and node not in coveraged_set ]
            # print("length set ", len(coveraged_set))
            random_id = random.choice(candidate_node_ids_clusters)
            selected_node = Node.nodes[random_id]
            temp_cluster_heads.append(selected_node)
            current_cluster_heads_ids.append(selected_node.id)

        graph_nodes = [self.network.sink_node] + self.network.available_nodes
        graph = Graph(graph_nodes, (self.network.R)/3) 
        connected,_ = graph.is_connected_with_component()
        if connected:
            # print("R/3 connected")
            graph_nodes = [self.network.sink_node] + temp_cluster_heads
            graph = Graph(graph_nodes, (self.network.R)/3)
            connected,_ = graph.is_connected_with_component()
            while not connected:
                candidate_node_ids_connect = [node.id for node in self.network.available_nodes if not node.is_sink and node not in temp_cluster_heads]
                if not candidate_node_ids_connect:
                    print("Bug rồi :<")
                    break  # Không còn node nào khả dụng

                random_id = random.choice(candidate_node_ids_connect)
                selected_node = Node.nodes[random_id]
                temp_cluster_heads.append(selected_node)

                # Tạo lại graph thay vì chỉ cập nhật nodes
                graph_nodes = [self.network.sink_node] + temp_cluster_heads
                graph = Graph(graph_nodes, (self.network.R)/3)  # Tạo lại đồ thị mới
                connected,_ = graph.is_connected_with_component()
            
        else:
            non_accept = 1
            graph_nodes = [self.network.sink_node] + temp_cluster_heads
            graph = Graph(graph_nodes, (self.network.R))
            connected, component = graph.is_connected_with_component()
            
            while not connected:
                candidate_node_ids_connect = [node.id for node in self.network.available_nodes if not node.is_sink and node not in temp_cluster_heads]
                if not candidate_node_ids_connect:  # Tránh lỗi khi không còn node khả dụng
                    print("Không còn node nào để kết nối mạng!")
                    break
                random_id = random.choice(candidate_node_ids_connect)
                selected_node = Node.nodes[random_id]
                graph.nodes.append(selected_node)
                temp_cluster_heads.append(selected_node)
                graph_nodes = [self.network.sink_node] + temp_cluster_heads
                graph = Graph(graph_nodes, (self.network.R))
                connected, component = graph.is_connected_with_component()
        
        return non_accept, temp_cluster_heads, current_cluster_heads_ids
        # self.network.cluster_heads_buffer.store(current_cluster_heads_ids)

def run(network, P, K1, K2, K, display=False):

    graph_nodes = [network.sink_node] + network.available_nodes
    graph = Graph(graph_nodes, (network.R))
    connected, component = graph.is_connected_with_component()
    # self.network.display_network()
    # print("component = ", len(component))
    if not connected:            
        # print("not connected!")
        network.available_nodes = [node for node in network.available_nodes if not node.is_sink and node.id in component]
    num_clus = math.ceil(len(network.available_nodes) / 10)
    fcm = FuzzyCMeansClustering(network, num_clusters=num_clus)
    non_accept, temp_cluster_heads, current_cluster_heads_ids = fcm.fit()
    # network.display_network(folder="new_log")
    if non_accept==1:
        energy_loss, is_k_connect, final_chs = network.step(temp_cluster_heads, non_accept, display)
    # network.display_network(folder="log_log")
    else:
        energy_loss, is_k_connect, final_chs = network.step(temp_cluster_heads, non_accept, display)    
        network.restep()
        # network.display_network(folder="new_log0")
        if is_k_connect == True:
            buffer.add(final_chs)
            # print("add : ", [node.id for node in final_chs], "len buffer  = ", len(buffer.history))
            # network.display_network(folder="check_check") # same input
            energy_loss, is_k_connect, final_chs = network.step(temp_cluster_heads, non_accept, display, folder="test")
            # network.display_network(folder="checkdcmm")
            current_energy = 0
            for node in network.available_nodes:
                current_energy += node.energy
            print("current_energy = ", current_energy )
            print("energy_loss = ", energy_loss)
        else:
            print("rechoose!!")
            history = buffer.get_history()
            candidates = []  # Danh sách lưu các candidate hợp lệ
            for candidate_chs in history:
                ids = [node.id for node in candidate_chs]
                # print("ids = ", ids )
                candidate_chs = [node for node in network.available_nodes if node.id in ids]
                candidate_cluster_heads_ids = []
                # print("test: ", [node.id for node in candidate_chs])
                for ch_nodes in candidate_chs:
                    candidate_cluster_heads_ids.append(ch_nodes.id)
                candidate_non_accept, candidate_temp_cluster_heads, candidate_cluster_heads_ids = fcm.select_more_cluster_heads(candidate_chs, candidate_cluster_heads_ids)
                candidate_energy_loss, candidate_is_k_connect, candidate_final_chs = network.step(candidate_temp_cluster_heads, candidate_non_accept, display, folder="candidate")
                network.restep()
                if candidate_is_k_connect == True:
                    print("Accepted")
                    # Lưu candidate hợp lệ cùng với energy_loss
                    candidates.append({
                        'chs': candidate_final_chs,
                        'energy_loss': candidate_energy_loss,
                        'non_accept': candidate_non_accept,
                        'temp_chs': candidate_temp_cluster_heads,
                    })
        
            # Lọc top 5 candidates có energy_loss thấp nhất
            top_3_candidates = sorted(candidates, key=lambda x: x['energy_loss'])[:3]
        
            # Chọn ngẫu nhiên một candidate từ top 5 (nếu có)
            if top_3_candidates:
                selected_candidate = random.choice(top_3_candidates)
                temp_final_chs = selected_candidate['temp_chs']
                final_non_accept = selected_candidate['non_accept']
                before_energy_loss = selected_candidate['energy_loss']
                candidate_energy_loss, candidate_is_k_connect, candidate_final_chs = network.step(temp_final_chs, final_non_accept, display, folder="test")
                print("Selected done : ", before_energy_loss, " ", candidate_energy_loss)
                current_energy = 0
                for node in network.available_nodes:
                    current_energy += node.energy
                print("current_energy = ", current_energy )
            else:
                print("No valid candidates found with is_k_connect == True")
                energy_loss, is_k_connect, final_chs = network.step(temp_cluster_heads, non_accept, display, folder="test")
                current_energy = 0
                for node in network.available_nodes:
                    current_energy += node.energy
                print("current_energy = ", current_energy )
                print("energy_loss = ", energy_loss)
    network.reset()

