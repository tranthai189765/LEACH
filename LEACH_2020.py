import math
from collections import defaultdict
import copy
from entity import Node
from graph_updated import Graph
import random

class ClusterHeadSelectorI_LEACH:
    def __init__(self, network):
        self.network = network

    def select_cluster_heads(self, p):
        # Bước 1: Chia mạng thành các cụm hình tròn (circular clusters)
        W = self.network.width
        H = self.network.height
        N = len([n for n in self.network.available_nodes if not n.is_sink])
        numRx = int(math.sqrt(p * N))  # số cụm trên mỗi chiều
        if numRx == 0:
            numRx = 1  # tránh chia cho 0

        dr = W / numRx  # đường kính mỗi cụm
        radius = dr / 2

        # Bước 2: Gom các nodes vào các cụm (vị trí center của cụm: lưới cách đều)
        clusters = defaultdict(list)
        for node in self.network.available_nodes:
            if node.is_sink:
                continue
            x = node.x
            y = node.y
            i = int(x // dr)
            j = int(y // dr)
            cluster_id = (i, j)
            clusters[cluster_id].append(node)

        # Bước 3: Tính năng lượng trung bình toàn mạng
        total_energy = sum(n.energy for n in self.network.available_nodes if not n.is_sink)
        avg_energy = total_energy / N if N > 0 else 0

        # Bước 4: Trong mỗi cụm, chọn 1 node có năng lượng cao hơn mức trung bình
        selected_chs = []
        selected_ch_ids = []

        for cluster_nodes in clusters.values():
            # lọc node đủ năng lượng
            valid_nodes = [n for n in cluster_nodes if n.energy >= avg_energy]
            if not valid_nodes:
                continue
            # chọn node có năng lượng cao nhất trong cụm
            ch_node = max(valid_nodes, key=lambda n: n.energy)
            selected_chs.append(ch_node)
            selected_ch_ids.append(ch_node.id)

        # lưu lại lịch sử các CHs
        self.network.cluster_heads_buffer.add(copy.deepcopy(selected_ch_ids))
        
        # Bạn có thể tùy chọn thêm bước "select_more_cluster_heads" nếu muốn
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
    non_accept, temp_cluster_heads, current_cluster_heads_ids = leach.select_cluster_heads(p = P)
    energy_loss, is_k_connect, final_chs = network.step(temp_cluster_heads, non_accept, display)
    for node in network.available_nodes:
        current_energy += node.energy
    print("current_energy = ", current_energy )
    print("energy_loss = ", energy_loss)
    network.reset()

