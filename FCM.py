import numpy as np
import skfuzzy as fuzz
from entity import Node 
from network import Network 
import random
import math
from chs_buffer import Buffer
import copy
import numpy as np
from graph_updated import Graph

# Initialize buffer
buffer = Buffer(10)

def euclidean_distances_np(a, b):
    return np.sqrt(((a[:, np.newaxis, :] - b[np.newaxis, :, :]) ** 2).sum(axis=2))

class FuzzyCMeansClustering:
    def __init__(self, network, num_clusters, m=2, max_iter=1e12, tol=1e-10):
        self.network = network
        self.num_clusters = num_clusters
        self.m = m  # Bậc mờ
        self.max_iter = max_iter
        self.tol = tol  # Ngưỡng hội tụ
        self.cluster_centers = None
        self.membership_matrix = None
        self.labels = None

    def fit(self):
        """Thực hiện phân cụm Fuzzy C-Means"""
        nodes = np.array([[node.x, node.y] for node in self.network.available_nodes if not node.is_sink])  # Lấy tọa độ nodes

        if nodes.shape[0] < self.num_clusters:
            print(f"[DEBUG] Số lượng node ({nodes.shape[0]}) nhỏ hơn số cluster ({self.num_clusters})")
            return

        if nodes.shape[0] == 0:
            print("[DEBUG] Không có node nào để phân cụm.")
            return

        # 1. Chạy thuật toán FCM
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            nodes.T, self.num_clusters, self.m, error=self.tol, maxiter=self.max_iter, init=None
        )

        # 2. Xác định mỗi node thuộc cụm nào (dựa vào giá trị membership cao nhất)
        cluster_labels = np.argmax(u, axis=0)

        # 3. Chọn Cluster Head (CH) theo tiêu chí hỗn hợp
        # self.select_cluster_heads(cluster_labels, cntr)

        # Lưu kết quả
        self.cluster_centers = cntr
        self.membership_matrix = u
        self.labels = cluster_labels
        print("donedone")

        # Select cluster heads considering R1 constraint
        non_accept, temp_cluster_heads, current_cluster_heads_ids = self.select_cluster_heads(self.labels, self.cluster_centers)
        return non_accept, temp_cluster_heads, current_cluster_heads_ids
    def select_cluster_heads(self, cluster_labels, cluster_centers):

        """Chọn Cluster Head theo tiêu chí 65% năng lượng, 35% khoảng cách đến tâm cụm"""
        temp_cluster_heads = []  # Reset cluster_heads
        max_energy = max(node.energy for node in self.network.available_nodes)

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
                energy_score = node.energy / max_energy  # Chuẩn hóa năng lượng về [0,1]
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
        max_energy = max(node.energy for node in self.network.available_nodes)
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
                for node in self.network.available_nodes:
                    if node.distance_to(self.network.sink_node) <= self.network.R:
                            temp_cluster_heads.append(node)
                            current_cluster_heads_ids.append(node.id)
            else:
                random_id = random.choice(candidate_node_ids_clusters)
                selected_node = Node.nodes[random_id]
                temp_cluster_heads.append(selected_node)
                current_cluster_heads_ids.append(selected_node.id)

        graph_nodes = [self.network.sink_node] + self.network.available_nodes
        graph = Graph(graph_nodes, (self.network.R)/3) 
        connected,_,_ = graph.is_connected_with_component()
        print("check len = ", len(temp_cluster_heads))
        if connected:
            # print("R/3 connected")
            # print("this")
            graph_nodes = [self.network.sink_node] + temp_cluster_heads
            graph = Graph(graph_nodes, (self.network.R)/3)
            connected,_,components = graph.is_connected_with_component()
            while not connected:
                # print("this")
                candidate_node_ids_connect = [node.id for node in self.network.available_nodes if not node.is_sink and node not in temp_cluster_heads]
                candidate_nodes = [node for node in self.network.available_nodes if not node.is_sink and node not in temp_cluster_heads]
                if not candidate_node_ids_connect:
                    print("Bug rồi :<")
                    break  # Không còn node nào khả dụng

                first, second = random.sample(range(len(components)), 2)
                min_dist = float("inf")
                bridge_x = bridge_y = None
                for node1_id in components[first]:
                    for node2_id in components[second]:
                        node1 = Node.nodes[node1_id]
                        node2 = Node.nodes[node2_id]
                        dist = math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            # Chọn node trung gian nằm giữa node1 và node2
                            bridge_x = (node1.x + node2.x) / 2
                            bridge_y = (node1.y + node2.y) / 2

                bridge_node = min(
                    candidate_nodes,
                    key=lambda x: math.sqrt((x.x - bridge_x)**2 + (x.y - bridge_y)**2)/1000
                )
                prob = random.uniform(0, 1)
                if(prob < 0.05):
                
                    temp_cluster_heads.append(bridge_node)
                    current_cluster_heads_ids.append(bridge_node.id)
                else:
                    random_id = random.choice(candidate_node_ids_connect)
                    selected_node = Node.nodes[random_id]
                    temp_cluster_heads.append(selected_node)
                    current_cluster_heads_ids.append(selected_node.id)

                # temp_cluster_heads.append(selected_node)

                # Tạo lại graph thay vì chỉ cập nhật nodes
                graph_nodes = [self.network.sink_node] + temp_cluster_heads
                graph = Graph(graph_nodes, (self.network.R)/3)  # Tạo lại đồ thị mới
                connected, _, components= graph.is_connected_with_component()
            
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
                graph_nodes = [self.network.sink_node] + temp_cluster_heads
                graph = Graph(graph_nodes, (self.network.R))
                connected, component, _ = graph.is_connected_with_component()
        print("check len after add bridget = ", len(temp_cluster_heads))
        return non_accept, temp_cluster_heads, current_cluster_heads_ids
        # self.network.cluster_heads_buffer.store(current_cluster_heads_ids)

def run(network, P, K1, K2, K, display=False):

    graph_nodes = [network.sink_node] + network.available_nodes
    graph = Graph(graph_nodes, (network.R))
    connected, component, _ = graph.is_connected_with_component()
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
        current_energy = 0
        energy_loss, is_k_connect, final_chs = network.step(temp_cluster_heads, non_accept, display)
        for node in network.available_nodes:
            current_energy += node.energy
        print("current_energy = ", current_energy )
        print("energy_loss = ", energy_loss)
    # network.display_network(folder="log_log")
    else:
        energy_loss, is_k_connect, final_chs = network.step(temp_cluster_heads, non_accept, display)    
        check = False
        prob = 1.0 
        for node in network.available_nodes:
            if(node.energy < 100):
                # print("Check lai")
                check = True
                break
        if check==True:
            prob = random.uniform(0, 1)
            print("prob = ", prob)
        network.restep()
        # network.display_network(folder="new_log0")
        if is_k_connect == True and prob > 0.1:
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
            if is_k_connect == True: buffer.add(final_chs)
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
        
            if candidates:
                losses = np.array([cand['energy_loss'] for cand in candidates])
                
                inv_losses = 1 / (losses + 1e-6)  # tránh chia cho 0
                sum_inv = np.sum(inv_losses)
                
                if sum_inv == 0 or np.isnan(sum_inv):
                    probs = np.ones(len(candidates)) / len(candidates)  # fallback
                else:
                    probs = inv_losses / sum_inv

                print("Danh sách losses:", losses.tolist())
                print("Sau khi softmax:", probs.tolist())

                selected_index = np.random.choice(len(candidates), p=probs)
                selected_candidate = candidates[selected_index]

                temp_final_chs = selected_candidate['temp_chs']
                final_non_accept = selected_candidate['non_accept']
                before_energy_loss = selected_candidate['energy_loss']
                candidate_energy_loss, candidate_is_k_connect, candidate_final_chs = network.step(temp_final_chs, final_non_accept, display, folder="test")
                print("Selected done : ", before_energy_loss, " ", candidate_energy_loss)
                current_energy = sum(node.energy for node in network.available_nodes)
                print("current_energy = ", current_energy)
            else:
                print("No valid candidates found with is_k_connect == True")
                energy_loss, is_k_connect, final_chs = network.step(temp_cluster_heads, non_accept, display, folder="test")
                current_energy = sum(node.energy for node in network.available_nodes)
                print("current_energy = ", current_energy)
                print("energy_loss = ", energy_loss)

    network.reset()