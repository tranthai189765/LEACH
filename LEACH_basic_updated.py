from entity import Node
from graph_updated import Graph
import random
import copy
time_out = 0
fixed_k = 0
non_accept = 0
class ClusterHeadSelector:
    def __init__(self, network):
        self.network = network

    def select_cluster_heads(self, P):
        maximum_round = int(1/P)
        last_cluster_heads = self.network.cluster_heads_buffer.take(maximum_round)
        print("last_cluster_heads : ", len(last_cluster_heads))
        candidate_nodes = [node for node in self.network.available_nodes 
                        if not node.is_sink and node.id not in last_cluster_heads]
        current_cluster_heads_ids = []
        temp_cluster_heads = []  # Reset cluster_heads
        for node in candidate_nodes:
            probability = P/(1 - P*(self.network.time_life % maximum_round))
            random_number = random.uniform(0, 1)
            if (random_number <= probability):
                temp_cluster_heads.append(node)
                current_cluster_heads_ids.append(node.id)
        print("len temp_cluster_heads = ", len(temp_cluster_heads))
        self.network.cluster_heads_buffer.add(copy.deepcopy(current_cluster_heads_ids))
        # print("last_cluster_heads : ", len(self.network.cluster_heads_buffer.take(maximum_round)))
        non_accept, temp_cluster_heads, current_cluster_heads_ids = self.select_more_cluster_heads(temp_cluster_heads, current_cluster_heads_ids)
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
    leach = ClusterHeadSelector(network)
    non_accept, temp_cluster_heads, current_cluster_heads_ids = leach.select_cluster_heads(P=P)
    energy_loss, is_k_connect, final_chs = network.step(temp_cluster_heads, non_accept, display)
    for node in network.available_nodes:
        current_energy += node.energy
    print("current_energy = ", current_energy )
    print("energy_loss = ", energy_loss)
    network.reset()
