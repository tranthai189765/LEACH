from entity import Node
from graph_updated import Graph
import random
def select_cluster_heads(network, P):
    maximum_round = int(1/P)
    last_cluster_heads = network.cluster_heads_buffer.take(maximum_round)
    print("last_cluster_heads : ", len(last_cluster_heads))
    candidate_nodes = [node for node in network.available_nodes 
                       if not node.is_sink and node.id not in last_cluster_heads]
    current_cluster_heads_ids = []
    for node in candidate_nodes:
        probability = P/(1 - P*(network.time_life % maximum_round))
        random_number = random.uniform(0, 1)
        if (random_number <= probability):
            node.is_cluster_head = True
            node.update_status()
            network.cluster_heads.append(node)
            network.cluster_infos.append([node.id])
            node.cluster_index = len(network.cluster_infos)
            current_cluster_heads_ids.append(node.id)
            network.add_edge(node.id, network.sink_node.id, "purple")

    network.cluster_heads_buffer.store(current_cluster_heads_ids)

def select_additional_cluster_heads(network):

    candidate_node_ids = [node.id for node in network.available_nodes if not node.is_sink and node.color != "red"]
    graph_nodes = []
    graph_nodes.append(network.sink_node)
    for cluster_head in network.cluster_heads:
        graph_nodes.append(cluster_head)
    graph = Graph(graph_nodes, network.R)
    while not graph.is_connected:
        random_id = random.choice(candidate_node_ids)
        selected_node = Node.nodes[random_id]
        selected_node.is_cluster_head = True
            node.update_status()
            network.cluster_heads.append(node)
            network.cluster_infos.append([node.id])
            node.cluster_index = len(network.cluster_infos)
            current_cluster_heads_ids.append(node.id)
            network.add_edge(node.id, network.sink_node.id, "purple")


def select_clusters(network):
    candidate_nodes = [node for node in network.available_nodes if not node.is_sink]
    for node in candidate_nodes:
        if node.color != "red":
            if len(network.cluster_heads) == 0:
                network.error = "No choice for cluster heads"
                break

            network.error = "None"
            closest_head = min(network.cluster_heads, key=lambda head: node.distance_to(head))
            network.cluster_infos[closest_head.cluster_index - 1].append(node.id)  # Thêm .id để giữ định dạng
            node.cluster_index = closest_head.cluster_index
            network.add_edge(node.id, closest_head.id, "lightblue")

def update_energy(network, K1, K2):
    #Update energy for cluster heads
    for ch in network.cluster_heads:
        distance = ch.distance_to(network.sink_node)
        ch.energy -= K1*distance
    
    for cluster in network.cluster_infos: 
        ch_id = cluster[0]
        cluster_head = Node.nodes[ch_id]
        for cm_id in cluster[1:]:
            cluster_member = Node.nodes[cm_id]
            if not cluster_member.is_sink:
                distance = cluster_member.distance_to(cluster_head) 
                cluster_member.energy -= distance * K2 
                cluster_head.energy -= distance * K2

def run(network, P, K1, K2, display=False):
    select_cluster_heads(network, P)
    select_clusters(network)
    if(display):
        network.display_network()
    update_energy(network, K1, K2)
    for node in network.available_nodes:
        node.update_all()
    network.update_network()
    network.save_network()
    for node in network.available_nodes:
        node.is_cluster_head = False
        node.update_status()
    network.time_life = network.time_life + 1
    network.reset()
