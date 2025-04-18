import numpy as np
import skfuzzy as fuzz
from entity import Node 
from network import Network 
import random
import math
from graph_updated import Graph
non_accept = 0
time_out = 0
fixed_k = 0
class FuzzyCMeansClustering:
    def __init__(self, network, num_clusters, m=2, max_iter=1e8, tol=1e-5):
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

        # 1. Chạy thuật toán FCM
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            nodes.T, self.num_clusters, self.m, error=self.tol, maxiter=self.max_iter, init=None
        )

        # 2. Xác định mỗi node thuộc cụm nào (dựa vào giá trị membership cao nhất)
        cluster_labels = np.argmax(u, axis=0)

        # 3. Chọn Cluster Head (CH) theo tiêu chí hỗn hợp
        self.select_cluster_heads(cluster_labels, cntr)

        # Lưu kết quả
        self.cluster_centers = cntr
        self.membership_matrix = u
        self.labels = cluster_labels

    def select_cluster_heads(self, cluster_labels, cluster_centers):
        """Chọn Cluster Head theo tiêu chí 65% năng lượng, 35% khoảng cách đến tâm cụm"""
        self.network.cluster_heads = []  # Reset cluster_heads

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
                return 0.75 * energy_score + 0.35 * distance_score

            # Chọn node có điểm cao nhất làm Cluster Head
            current_cluster_heads_ids = []
            ch_id = max(cluster_nodes, key=score)
            ch_node = Node.nodes[ch_id]
            
            # Lưu vào danh sách Cluster Heads
            ch_node.is_cluster_head = True
            ch_node.update_status()
            self.network.cluster_heads.append(ch_node)
            self.network.cluster_infos.append([ch_node.id])
            ch_node.cluster_index = len(self.network.cluster_infos)
            current_cluster_heads_ids.append(ch_node.id)
    
        while not self.network.is_coveraged()[1]:
            coveraged_set, _ = self.network.is_coveraged()
            candidate_node_ids_clusters = [node.id for node in self.network.available_nodes if not node.is_sink and node not in coveraged_set ]
            # print("length set ", len(coveraged_set))
            random_id = random.choice(candidate_node_ids_clusters)
            selected_node = Node.nodes[random_id]
            selected_node.is_cluster_head = True
            selected_node.update_status()
            self.network.cluster_heads.append(selected_node)
            self.network.cluster_infos.append([selected_node.id])
            selected_node.cluster_index = len(self.network.cluster_infos) 
            current_cluster_heads_ids.append(selected_node.id)

        
        graph_nodes = [self.network.sink_node] + self.network.available_nodes
        graph = Graph(graph_nodes, (self.network.R)/3) 
        connected,_ = graph.is_connected_with_component()
        if connected:
            graph_nodes = [self.network.sink_node] + self.network.cluster_heads
            graph = Graph(graph_nodes, (self.network.R)/3)
            connected,_ = graph.is_connected_with_component()
            while not connected:
                candidate_node_ids_connect = [node.id for node in self.network.available_nodes if not node.is_sink and node not in self.network.cluster_heads]
                if not candidate_node_ids_connect:
                    print("Bug rồi :<")
                    break  # Không còn node nào khả dụng

                random_id = random.choice(candidate_node_ids_connect)
                selected_node = Node.nodes[random_id]
                    
                # Cập nhật node mới là cluster head
                selected_node.is_cluster_head = True
                selected_node.update_status()
                self.network.cluster_heads.append(selected_node)
                self.network.cluster_infos.append([selected_node.id])
                selected_node.cluster_index = len(self.network.cluster_infos)

                # Tạo lại graph thay vì chỉ cập nhật nodes
                graph_nodes = [self.network.sink_node] + self.network.cluster_heads
                graph = Graph(graph_nodes, (self.network.R)/3)  # Tạo lại đồ thị mới
                connected,_ = graph.is_connected_with_component()
            
        else:
            global non_accept
            global time_out
            non_accept = 1
            graph_nodes = [self.network.sink_node] + self.network.available_nodes
            graph = Graph(graph_nodes, (self.network.R))
            connected, component = graph.is_connected_with_component()
            # self.network.display_network()
            if not connected:
                for node in self.network.available_nodes:
                    if node.id not in component:
                        # node.is_dead = True
                        # node.update_status()
                        # print("do nothing")
                        break
                
                # Lọc
                if len(component)==1:
                    time_out = 1
                    # self.network.display_network()
                    # print("Không kết nối được sink")
                # Giữ lại chỉ các node trong thành phần liên thông với sink
                self.network.available_nodes = [node for node in self.network.available_nodes if not node.is_sink and node.id in component]
                self.network.cluster_heads = [node for node in self.network.cluster_heads if not node.is_sink and node.id in component]
                # print("Change!!")
                # print("num nodes = ", len(self.network.available_nodes))
                # self.network.display_network()
            graph_nodes = [self.network.sink_node] + self.network.cluster_heads
            graph = Graph(graph_nodes, (self.network.R))
            connected, component = graph.is_connected_with_component()
            
            while not connected and time_out==0:
                candidate_node_ids_connect = [node.id for node in self.network.available_nodes if not node.is_sink and node not in self.network.cluster_heads]
                if not candidate_node_ids_connect:  # Tránh lỗi khi không còn node khả dụng
                    print("Không còn node nào để kết nối mạng!")
                    break
                random_id = random.choice(candidate_node_ids_connect)
                selected_node = Node.nodes[random_id]
                graph.nodes.append(selected_node)
                selected_node.is_cluster_head = True
                selected_node.update_status()
                self.network.cluster_heads.append(selected_node)
                self.network.cluster_infos.append([selected_node.id])
                selected_node.cluster_index = len(self.network.cluster_infos) 
                graph_nodes = [self.network.sink_node] + self.network.cluster_heads
                graph = Graph(graph_nodes, (self.network.R))
                connected, component = graph.is_connected_with_component()
        self.network.cluster_heads_buffer.store(current_cluster_heads_ids)

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
            else:
                print("buggg")


def update_energy(network, K1, K2):

    graph_nodes = []
    graph_nodes.append(network.sink_node)
    for cluster_head in network.cluster_heads:
        graph_nodes.append(cluster_head)
    global fixed_k 
    graph_chs = None
    global non_accept 
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
    global fixed_k
    global non_accept
    num_clus = math.ceil(len(network.available_nodes) / 10)
    fcm = FuzzyCMeansClustering(network, num_clusters=num_clus)
    fcm.fit()
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


