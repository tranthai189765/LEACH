""" Simulation """
import random
from graph import Graph
from entity import Node
def random_cluster_head(network, num_clusters):
    # Lọc ra các node không phải là sink
    candidate_nodes = [node for node in network.available_nodes if not node.is_sink]
    
    # Kiểm tra nếu số lượng node hợp lệ ít hơn num_clusters
    if len(candidate_nodes) < num_clusters:
        raise ValueError("Số lượng node hợp lệ ít hơn số cluster yêu cầu.")
    
    # Chọn ngẫu nhiên num_clusters node từ danh sách hợp lệ
    selected_nodes = random.sample(candidate_nodes, num_clusters)
    
    # Đặt thuộc tính is_cluster_head thành True cho các node được chọn
    for node in selected_nodes:
        node.is_cluster_head = True
        node.update_status()
        network.cluster_heads.append(node)
        network.cluster_infos.append([node.id])
        node.cluster_index = len(network.cluster_infos)
        
    return selected_nodes

def select_clusters(network):
    candidate_nodes = [node for node in network.available_nodes if not node.is_sink]
    for node in candidate_nodes:
        if node.color != "red":  
            closest_head = min(network.cluster_heads, key=lambda head: node.distance_energy_to(head))
            network.cluster_infos[closest_head.cluster_index - 1].append(node.id)  # Thêm .id để giữ định dạng
            node.cluster_index = closest_head.cluster_index
            network.add_edge(node.id, closest_head.id, "lightblue")

def connect_clusters(network):
    graph_nodes = []
    graph_nodes.append(network.sink_node)
    for cluster_head in network.cluster_heads:
        graph_nodes.append(cluster_head)
    graph = Graph(graph_nodes)
    mst, total_weight = graph.prim_mst()
    for id1, id2 in mst:
        network.add_edge(id1, id2, "purple")
    
    return total_weight

def update_energy(network, K1, K2, total_weight):
    #Update energy for cluster heads
    for ch in network.cluster_heads:
        ch.energy -= K1*total_weight
    
    for cluster in network.cluster_infos: 
        ch_id = cluster[0]
        cluster_head = Node.nodes[ch_id]
        for cm_id in cluster[1:]:
            cluster_member = Node.nodes[cm_id]
            if not cluster_member.is_sink:
                distance = cluster_member.distance_to(cluster_head) 
                cluster_member.energy -= distance * K2 
                cluster_head.energy -= distance * K2

def run(network, number_clusters, K1, K2, display=False):
    random_cluster_head(network, number_clusters)
    select_clusters(network)
    total_weight = connect_clusters(network)
    if(display):
        network.display_network()
    update_energy(network, K1, K2, total_weight)
    for node in network.available_nodes:
        node.update_all()
    network.update_network()
    network.save_network()
    for node in network.available_nodes:
        node.is_cluster_head = False
        node.update_status()
    network.reset()


