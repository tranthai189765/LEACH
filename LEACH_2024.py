import math
import copy
import random
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans 

from entity import Node             
from graph_updated import Graph
class ClusterHeadSelectorI_LEACH:
    def __init__(self, network):
        self.network = network

    def select_cluster_heads(self, k_clusters):
        # Bước 1: Lấy tất cả node không phải sink
        nodes = [n for n in self.network.available_nodes if not n.is_sink]
        if len(nodes) < k_clusters:
            k_clusters = max(1, len(nodes))

        positions = np.array([[n.x, n.y] for n in nodes])
        node_ids = [n.id for n in nodes]

        # Bước 2: Phân cụm bằng KMeans
        kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10).fit(positions)
        labels = kmeans.labels_

        selected_chs = []
        selected_ch_ids = []

        for cluster_idx in range(k_clusters):
            cluster_nodes = [nodes[i] for i in range(len(nodes)) if labels[i] == cluster_idx]

            # Tính tổng khoảng cách từ mỗi node đến tất cả node khác trong cụm
            def total_distance(candidate_node):
                return sum(
                    np.linalg.norm(np.array([candidate_node.x, candidate_node.y]) - np.array([other.x, other.y]))
                    for other in cluster_nodes
                )

            # Chọn node có tổng khoảng cách nhỏ nhất → làm CH
            ch_node = min(cluster_nodes, key=total_distance)
            selected_chs.append(ch_node)
            selected_ch_ids.append(ch_node.id)

        # Lưu lịch sử CH
        self.network.cluster_heads_buffer.add(copy.deepcopy(selected_ch_ids))

        # Xử lý nếu chưa đủ cover
        non_accept, selected_chs, selected_ch_ids = self.select_more_cluster_heads(selected_chs, selected_ch_ids)
        return non_accept, selected_chs, selected_ch_ids

    
    def select_more_cluster_heads(self, temp_cluster_heads, current_cluster_heads_ids):
        non_accept = 0
        while True:
            coveraged_set, is_fully_coveraged = self.network.is_coveraged(temp_cluster_heads)
            if is_fully_coveraged:
                break
        
            coveraged_node_ids = {node.id for node in coveraged_set}
            candidate_node_ids_clusters = [
                node.id for node in self.network.available_nodes 
                if not node.is_sink and node.id not in coveraged_node_ids
            ]
            if not candidate_node_ids_clusters:
                break
            else:
                random_id = random.choice(candidate_node_ids_clusters)
                selected_node = Node.nodes[random_id]
                temp_cluster_heads.append(selected_node)
                current_cluster_heads_ids.append(selected_node.id)

        graph_nodes = [self.network.sink_node] + self.network.available_nodes
        graph = Graph(graph_nodes, (self.network.R)/3) 
        connected,_,_ = graph.is_connected_with_component()
        if connected:
            # print("R/3 connected")
            graph_nodes = [self.network.sink_node] + temp_cluster_heads
            graph = Graph(graph_nodes, (self.network.R)/3)
            connected,_,_ = graph.is_connected_with_component()
            while not connected:
                candidate_node_ids_connect = [node.id for node in self.network.available_nodes if not node.is_sink and node not in temp_cluster_heads]
                if not candidate_node_ids_connect:
                    print("Bug rồi :<")
                    break  # Không còn node nào khả dụng

                random_id = random.choice(candidate_node_ids_connect)
                selected_node = Node.nodes[random_id]
                temp_cluster_heads.append(selected_node)
                current_cluster_heads_ids.append(selected_node.id)

                # Tạo lại graph thay vì chỉ cập nhật nodes
                graph_nodes = [self.network.sink_node] + temp_cluster_heads
                graph = Graph(graph_nodes, (self.network.R)/3)  # Tạo lại đồ thị mới
                connected,_,_ = graph.is_connected_with_component()
            
        else:
            non_accept = 1
            graph_nodes = [self.network.sink_node] + temp_cluster_heads
            graph = Graph(graph_nodes, (self.network.R))
            connected, component,_ = graph.is_connected_with_component()
            
            while not connected:
                candidate_node_ids_connect = [node.id for node in self.network.available_nodes if not node.is_sink and node not in temp_cluster_heads]
                if not candidate_node_ids_connect:  # Tránh lỗi khi không còn node khả dụng
                    print("Không còn node nào để kết nối mạng!")
                    break
                random_id = random.choice(candidate_node_ids_connect)
                selected_node = Node.nodes[random_id]
                graph.nodes.append(selected_node)
                temp_cluster_heads.append(selected_node)
                current_cluster_heads_ids.append(selected_node.id)
                graph_nodes = [self.network.sink_node] + temp_cluster_heads
                graph = Graph(graph_nodes, (self.network.R))
                connected, component,_ = graph.is_connected_with_component()
        return non_accept, temp_cluster_heads, current_cluster_heads_ids


def run(network, P, K1, K2, K, display=False):
    graph_nodes = [network.sink_node] + network.available_nodes
    graph = Graph(graph_nodes, (network.R))
    connected, component, _ = graph.is_connected_with_component()
    current_energy = 0
    if not connected:            
        network.available_nodes = [node for node in network.available_nodes if not node.is_sink and node.id in component]
    leach = ClusterHeadSelectorI_LEACH(network)
    non_accept, temp_cluster_heads, current_cluster_heads_ids = leach.select_cluster_heads(k_clusters=10)
    energy_loss, is_k_connect, final_chs = network.step(temp_cluster_heads, non_accept, display)
    for node in network.available_nodes:
        current_energy += node.energy
    print("current_energy = ", current_energy )
    print("energy_loss = ", energy_loss)
    network.reset()

