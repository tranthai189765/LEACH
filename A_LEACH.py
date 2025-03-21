from entity import Node
import random
def select_cluster_heads_2(network, M):
    current_cluster_heads_2_ids = []
    for node in network.take_top(M):
        node.is_cluster_head_2 = True
        node.update_status()
        network.cluster_heads_2.append(node)
        network.cluster_2_infos.append([node.id])
        node.cluster_index = len(network.cluster_2_infos)
        current_cluster_heads_2_ids.append(node.id)
        network.add_edge(node.id, network.sink_node.id, "purple")

    network.cluster_heads_2_buffer.store(current_cluster_heads_2_ids)
    if(network.time_life == 10):
        print("Number cluster heads 2: ",len(network.cluster_heads_2))

def select_cluster_heads(network, P):
    maximum_round = int(1/P)
    last_cluster_heads = network.cluster_heads_buffer.take(maximum_round)
    print("last_cluster_heads : ", len(last_cluster_heads))
    candidate_nodes = [node for node in network.available_nodes 
                       if not node.is_sink and node.id not in last_cluster_heads and node not in network.cluster_heads_2]
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

    network.cluster_heads_buffer.store(current_cluster_heads_ids)

def select_clusters_2(network):
    candidate_nodes = [node for node in network.cluster_heads if not node.is_sink]
    if(network.time_life == 10):
                print("Number cluster heads 2 test 1: ",len(network.cluster_heads_2))
    for node in candidate_nodes:
        if node.color != "yellow":
            if(network.time_life == 10):
                print("Number cluster heads 2 test 2: ",len(network.cluster_heads_2))
            if len(network.cluster_heads_2) == 0:
                network.error = "No choice for cluster heads 2"
                break

            closest_head = min(network.cluster_heads_2, key=lambda head: node.distance_to(head))
            network.cluster_2_infos[closest_head.cluster_2_index - 1].append(node.id)  # Thêm .id để giữ định dạng
            node.cluster_2_index = closest_head.cluster_2_index
            network.add_edge(node.id, closest_head.id, "pink")


def select_clusters(network):
    candidate_nodes = [node for node in network.available_nodes if not node.is_sink]
    for node in candidate_nodes:
        if node.color != "red" and node.color != "yellow":
            if len(network.cluster_heads) == 0:
                network.error = "No choice for cluster heads"
                break

            closest_head = min(network.cluster_heads, key=lambda head: node.distance_to(head))
            network.cluster_infos[closest_head.cluster_index - 1].append(node.id)  # Thêm .id để giữ định dạng
            node.cluster_index = closest_head.cluster_index
            network.add_edge(node.id, closest_head.id, "lightblue")

def update_energy(network, K1, K2):
    #Update energy for cluster heads
    for ch_2 in network.cluster_heads_2:
        distance = ch_2.distance_to(network.sink_node)
        ch_2.energy -= K1*distance
    
    for cluster_2 in network.cluster_2_infos: 
        ch_2_id = cluster_2[0]
        cluster_head_2 = Node.nodes[ch_2_id]
        for ch_id in cluster_2[1:]:
            cluster_head = Node.nodes[ch_id]
            if not cluster_head.is_sink:
                distance = cluster_head.distance_to(cluster_head_2) 
                cluster_head.energy -= distance * K2 
                cluster_head_2.energy -= distance * K2
    
    for cluster in network.cluster_infos: 
        ch_id = cluster[0]
        cluster_head = Node.nodes[ch_id]
        for cm_id in cluster[1:]:
            cluster_member = Node.nodes[cm_id]
            if not cluster_member.is_sink:
                distance = cluster_member.distance_to(cluster_head) 
                cluster_member.energy -= distance * K2 
                cluster_head.energy -= distance * K2

def run(network, M, P, K1, K2, display=False):
    select_cluster_heads_2(network, M)
    select_cluster_heads(network, P)
    select_clusters_2(network)
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
        node.is_cluster_head_2 = False
        node.update_status()
    network.time_life = network.time_life + 1
    network.reset()
