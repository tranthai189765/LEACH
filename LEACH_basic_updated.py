from entity import Node
from graph_updated import Graph
import random
time_out = 0
fixed_k = 0
non_accept = 0
def select_cluster_heads(network, P):
    maximum_round = int(1/P)
    last_cluster_heads = network.cluster_heads_buffer.take(maximum_round)
    # print("last_cluster_heads : ", len(last_cluster_heads))
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
    
    # network.display_network()
    while not network.is_coveraged()[1]:
        coveraged_set, _ = network.is_coveraged()
        candidate_node_ids_clusters = [node.id for node in network.available_nodes if node not in coveraged_set ]
        # print("length set ", len(coveraged_set))
        random_id = random.choice(candidate_node_ids_clusters)
        selected_node = Node.nodes[random_id]
        selected_node.is_cluster_head = True
        selected_node.update_status()
        network.cluster_heads.append(selected_node)
        network.cluster_infos.append([selected_node.id])
        selected_node.cluster_index = len(network.cluster_infos) 
        current_cluster_heads_ids.append(selected_node.id)

    
    graph_nodes = [network.sink_node] + network.available_nodes
    graph = Graph(graph_nodes, (network.R)/3) 
    connected,_ = graph.is_connected_with_component()
    if connected:
        graph_nodes = [network.sink_node] + network.cluster_heads
        graph = Graph(graph_nodes, (network.R)/3)
        connected,_ = graph.is_connected_with_component()
        while not connected:
            candidate_node_ids_connect = [node.id for node in network.available_nodes if not node.is_sink and node not in network.cluster_heads]
            if not candidate_node_ids_connect:
                print("Bug rồi :<")
                break  # Không còn node nào khả dụng

            random_id = random.choice(candidate_node_ids_connect)
            selected_node = Node.nodes[random_id]
                
            # Cập nhật node mới là cluster head
            selected_node.is_cluster_head = True
            selected_node.update_status()
            network.cluster_heads.append(selected_node)
            network.cluster_infos.append([selected_node.id])
            selected_node.cluster_index = len(network.cluster_infos)

            # Tạo lại graph thay vì chỉ cập nhật nodes
            graph_nodes = [network.sink_node] + network.cluster_heads
            graph = Graph(graph_nodes, (network.R)/3)  # Tạo lại đồ thị mới
            connected,_ = graph.is_connected_with_component()
        
    else:
        global non_accept
        global time_out
        non_accept = 1
        graph_nodes = [network.sink_node] + network.available_nodes
        graph = Graph(graph_nodes, (network.R))
        connected, component = graph.is_connected_with_component()
        # network.display_network()
        if not connected:
            for node in network.available_nodes:
                if node.id not in component:
                    break
                    # node.is_dead = True
                    # node.update_status()
            
            # Lọc
            if len(component)==1:
                time_out = 1
                network.display_network()
                # print("Không kết nối được sink")
            # Giữ lại chỉ các node trong thành phần liên thông với sink
            network.available_nodes = [node for node in network.available_nodes if not node.is_sink and node.id in component]
            network.cluster_heads = [node for node in network.cluster_heads if not node.is_sink and node.id in component]
            # print("Change!!")
            # print("com = ", component)
            # network.display_network()
        graph_nodes = [network.sink_node] + network.cluster_heads
        graph = Graph(graph_nodes, (network.R))
        connected, component = graph.is_connected_with_component()
        
        while not connected and time_out==0:
            candidate_node_ids_connect = [node.id for node in network.available_nodes if not node.is_sink and node not in network.cluster_heads]
            if not candidate_node_ids_connect:  # Tránh lỗi khi không còn node khả dụng
                print("Không còn node nào để kết nối mạng!")
                break
            random_id = random.choice(candidate_node_ids_connect)
            selected_node = Node.nodes[random_id]
            graph.nodes.append(selected_node)
            selected_node.is_cluster_head = True
            selected_node.update_status()
            network.cluster_heads.append(selected_node)
            network.cluster_infos.append([selected_node.id])
            selected_node.cluster_index = len(network.cluster_infos) 
            graph_nodes = [network.sink_node] + network.cluster_heads
            graph = Graph(graph_nodes, (network.R))
            connected, component = graph.is_connected_with_component()
    network.cluster_heads_buffer.store(current_cluster_heads_ids)

def select_clusters(network, k):
    candidate_nodes = [node for node in network.available_nodes if not node.is_sink]
    for node in candidate_nodes:
        if node.color != "red":
            if len(network.cluster_heads) == 0:
                network.error = "No choice for cluster heads"
                break

            network.error = "None"
            valid_heads = [head for head in network.cluster_heads if head.degree < k]
            closest_head = min(valid_heads, key=lambda head: node.distance_to(head)) if valid_heads else min(network.cluster_heads, key=lambda head: node.distance_to(head))
            if node.distance_to(closest_head) > network.R: 
                closest_head =min(network.cluster_heads, key=lambda head: node.distance_to(head))

            network.cluster_infos[closest_head.cluster_index - 1].append(node.id)  # Thêm .id để giữ định dạng
            node.cluster_index = closest_head.cluster_index
            if node.distance_to(closest_head) <= network.R: 
                network.add_edge(node.id, closest_head.id, "lightblue")
                closest_head.degree += 1


def update_energy(network, K1, K2):

    graph_nodes = []
    graph_nodes.append(network.sink_node)
    for cluster_head in network.cluster_heads:
        graph_nodes.append(cluster_head)
    global non_accept 
    graph_chs = None
    if(non_accept == 0):
        graph_chs = Graph(graph_nodes, (network.R)/3)
    if(non_accept == 1):
        graph_chs = Graph(graph_nodes, (network.R))
    # print("is connected ? ", graph_chs.is_connected())
    mst = graph_chs.find_mst()
    for id1, id2 in mst:
        network.add_edge(id1, id2, "purple")
        node1 = Node.nodes[id1]
        node2 = Node.nodes[id2]
        if node1.is_sink == False: 
            node1.energy -= K1*node1.distance_to(node2)

        if node2.is_sink == False: 
            node2.energy -= K1*node1.distance_to(node2)
    
    for cluster in network.cluster_infos: 
        ch_id = cluster[0]
        cluster_head = Node.nodes[ch_id]
        for cm_id in cluster[1:]:
            cluster_member = Node.nodes[cm_id]
            if not cluster_member.is_sink:
                distance = cluster_member.distance_to(cluster_head) 
                cluster_member.energy -= distance * K2 
                cluster_head.energy -= distance * K2

def run(network, P, K1, K2, K, display=False):
    global time_out
    global non_accept
    select_cluster_heads(network, P)
    if time_out == 1:
        print("done!")
        network.is_dead = True
        return  # Cần đảm bảo đoạn code này nằm trong vòng lặ
    select_clusters(network, K)
    update_energy(network, K1, K2)
    if non_accept == 0:
        if network.is_k_connect(K):
            network.time_k_connect += 1
    non_accept = 0
    if(display):
        network.display_network()
    for node in network.available_nodes:
        node.update_all()
    network.update_network()
    network.save_network()
    for node in network.available_nodes:
        node.is_cluster_head = False
        node.degree = 1
        node.update_status()
    network.time_life = network.time_life + 1
    network.reset()
