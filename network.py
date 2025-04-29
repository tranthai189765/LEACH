"""Network Setup"""
W = 1000
H = 1000
from entity import Node 
import matplotlib.pyplot as plt
import copy  # Thêm dòng này để import module copy
import random
import json
from datetime import datetime
from buffer import ClusterHeadBufferTest
from graph_updated import Graph
class Network:
    def __init__(self, num_nodes, width=W, height=H, seed=None, K1=0.1, K2=0.05, K=2):
        self.width = width
        self.height = height
        self.num_nodes = num_nodes
        self.num_nodes_history = []
        self.seed = seed  # Lưu seed để có thể sử dụng lại
        self.nodes = self._generate_nodes()
        self.available_nodes = self.nodes
        self.sink_node = self._select_sink()
        self.cluster_heads = []
        self.cluster_heads_2 = []
        self.cluster_infos = []
        self.cluster_2_infos = []
        self.edges = []
        self.edge_colors = {}  # Dictionary to store edge colors
        self.time_life = 0
        self.cluster_heads_buffer = ClusterHeadBufferTest(mem_size=20)
        self.cluster_heads_2_buffer = ClusterHeadBufferTest(mem_size=20)
        self.error = "None"
        self.R = 300
        self.time_k_connect = 0
        self.is_dead = False
        self.K1 = K1
        self.K2 = K2
        self.K = K
        self._state_backup = None

    def _generate_nodes(self):
        """Sinh ngẫu nhiên các node trong miền (0,0) đến (width, height) với seed cố định"""
        if self.seed is not None:
            random.seed(self.seed)  # Đặt seed để tạo kết quả nhất quán
        return [Node(random.uniform(0, self.width), random.uniform(0, self.height)) for _ in range(self.num_nodes)]
    
    def _select_sink(self):
        """Chọn ngẫu nhiên một node làm sink"""
        sink = random.choice(self.nodes)
        sink.is_sink = True
        sink.update_status()
        return sink
    
    def is_coveraged(self, temp_cluster_heads):
        coveraged_nodes = set()
        for ch in temp_cluster_heads:
            for node in self.available_nodes:
                if ch.distance_to(node) <= self.R:
                    coveraged_nodes.add(node)
        is_fully_coveraged = len(coveraged_nodes) == len(self.available_nodes)
        # print("result = ", [node.id for node in self.available_nodes if node not in coveraged_nodes])
        return coveraged_nodes, is_fully_coveraged
    
    def is_k_connect(self, k):
        for ch in self.cluster_heads:
            if ch.degree < k:
                return False
        return True 

    
    def add_edge(self, source_id, target_id, color='black'):
        """Thêm các cạnh dựa trên id"""
        self.edges.append((source_id, target_id))
        self.edge_colors[(source_id, target_id)] = color
    
    def reset(self):
        """ Reset """
        self.cluster_heads = []
        self.cluster_heads_2 = []
        self.cluster_infos = []
        self.cluster_2_infos = []
        self.edges = []
        self.edge_colors = {}  # Dictionary to store edge colors
        self.error = "None"


    def _backup_state(self):
        """Lưu trạng thái đơn giản mà không copy object"""
        self._state_backup = {
            'time_life': self.time_life,
            'num_nodes_history': list(self.num_nodes_history),  # shallow copy là đủ
            'is_dead': self.is_dead,
            'error': self.error,
            'cluster_heads_ids': [node.id for node in self.cluster_heads],
            'cluster_infos': [list(cluster) for cluster in self.cluster_infos],
            'edges': list(self.edges),
            'edge_colors': dict(self.edge_colors),
            'nodes_state': {
                node.id: (
                    node.energy,
                    node.is_cluster_head,
                    node.degree,
                    node.cluster_index,
                    node.is_dead,
                    node.color
                )
                for node in self.nodes
            },
            'time_k_connect': self.time_k_connect
        }

    def restep(self):
        if self._state_backup is None:
            raise ValueError("No state backup available to restore.")
        
        # Khôi phục các thông tin tổng quan
        self.time_life = self._state_backup['time_life']
        self.num_nodes_history = list(self._state_backup['num_nodes_history'])
        self.is_dead = self._state_backup['is_dead']
        self.error = self._state_backup['error']
        self.cluster_infos = [list(cluster) for cluster in self._state_backup['cluster_infos']]
        self.edges = list(self._state_backup['edges'])
        self.edge_colors = dict(self._state_backup['edge_colors'])
        self.time_k_connect = self._state_backup['time_k_connect']
        
        # Khôi phục trạng thái từng node
        for node in self.nodes:
            state = self._state_backup['nodes_state'].get(node.id)
            if state:
                node.energy, node.is_cluster_head, node.degree, node.cluster_index, node.is_dead, node.color = state
                node.update_status()
        
        # Khôi phục cluster heads từ id
        self.cluster_heads = [Node.nodes[node_id] for node_id in self._state_backup['cluster_heads_ids']]

        # Đồng bộ hóa available_nodes (nếu cần)
        self.available_nodes = [node for node in self.nodes if not node.is_dead]
        
    def step(self, selected_chs, non_accept, display, folder="image"):

        # Đặt lại trạng thái của tất cả các node trước khi bắt đầu
        self.reset()
        # self.update_network()
        for node in self.available_nodes:
            node.is_cluster_head = False
            node.degree = 1  # Đặt lại degree
            node.cluster_index = None  # Đặt lại cluster_index
            node.update_status()
        # self.display_network(folder="new")
        energy_loss = 0
        is_k_connect = False
        final_chs = []
        # Lưu trạng thái trước khi thực hiện step
        self._backup_state()
        self.cluster_heads = []  # Reset cluster_heads
        if self.is_dead == True:
            print("done!")
            return  # Cần đảm bảo đoạn code này nằm trong vòng lặp
        for selected_node in selected_chs:
            selected_node.is_cluster_head = True
            selected_node.update_status()
            self.cluster_heads.append(selected_node)
            self.cluster_infos.append([selected_node.id])
            selected_node.cluster_index = len(self.cluster_infos) 
        
        print("number of cluster heads = ", len(self.cluster_heads))
        # print("last_cluster_heads : ", len(self.cluster_heads_buffer.take(10)))
        # Cluster members select clusters
        chs_loss = 0
        candidate_nodes = [node for node in self.available_nodes if not node.is_sink]
        for node in candidate_nodes:
            if node.color != "red":
                if len(self.cluster_heads) == 0:
                    self.error = "No choice for cluster heads"
                    break

                self.error = "None"
                valid_heads = [head for head in self.cluster_heads if head.degree < self.K]
                closest_head = min(valid_heads, key=lambda head: node.distance_to(head)) if valid_heads else min(self.cluster_heads, key=lambda head: node.distance_to(head))
                if node.distance_to(closest_head) > self.R: 
                    closest_head =min(self.cluster_heads, key=lambda head: node.distance_to(head))

                self.cluster_infos[closest_head.cluster_index - 1].append(node.id)  # Thêm .id để giữ định dạng
                node.cluster_index = closest_head.cluster_index
                if node.distance_to(closest_head) <= self.R: 
                    self.add_edge(node.id, closest_head.id, "lightblue")
                    closest_head.degree += 1
        
        graph_nodes = []
        graph_nodes.append(self.sink_node)
        for cluster_head in self.cluster_heads:
            graph_nodes.append(cluster_head)
        graph_chs = None
        if(non_accept == 0):
            graph_chs = Graph(graph_nodes, (self.R)/3)
        if(non_accept == 1):
            graph_chs = Graph(graph_nodes, (self.R))
        # print("is connected ? ", graph_chs.is_connected())
        sink_node  = self.sink_node
        mst = graph_chs.find_mst()
        if non_accept == 0 and self.is_k_connect(self.K):
            for id1, id2 in mst:
                self.add_edge(id1, id2, "purple")
                node1 = Node.nodes[id1]
                node2 = Node.nodes[id2]
                d = node1.distance_to(node2)

                # Mặc định cả 2 đều truyền (tốn năng lượng theo khoảng cách)
                # nhưng sẽ xét lại nếu 1 trong 2 là "nút nhận" trên đường đi tới sink

                # Kiểm tra nếu node2 nằm trên đường từ node1 đến sink => node2 là "nút nhận"
                if not node1.is_sink:
                    if graph_chs.is_on_path_to_sink(id1, id2, sink_node.id):
                        node2.energy -= self.K2  # Nhận => tốn ít
                        energy_loss += self.K2
                        chs_loss += self.K2
                    else:
                        node1.energy -= self.K1 * d
                        energy_loss += self.K1 * d
                        chs_loss += self.K1 * d

                # Kiểm tra nếu node1 nằm trên đường từ node2 đến sink => node1 là "nút nhận"
                if not node2.is_sink:
                    if graph_chs.is_on_path_to_sink(id2, id1, sink_node.id):
                        node1.energy -= self.K2  # Nhận => tốn ít
                        energy_loss += self.K2
                        chs_loss += self.K2
                    else:
                        node2.energy -= self.K1 * d
                        energy_loss += self.K1 * d
                        chs_loss += self.K1 * d
            
            print("chs_loss = ",chs_loss)
            for cluster in self.cluster_infos: 
                ch_id = cluster[0]
                cluster_head = Node.nodes[ch_id]
                for cm_id in cluster[1:]:
                    cluster_member = Node.nodes[cm_id]
                    if not cluster_member.is_sink:
                        distance = cluster_member.distance_to(cluster_head) 
                        cluster_member.energy -= distance * self.K1 
                        cluster_head.energy -= self.K2
                        energy_loss += distance * self.K1 + self.K2
        else:
            print("day ne")
            for id1, id2 in mst:
                self.add_edge(id1, id2, "purple")
                node1 = Node.nodes[id1]
                node2 = Node.nodes[id2]
                d = node1.distance_to(node2)

                # Mặc định cả 2 đều truyền (tốn năng lượng theo khoảng cách)
                # nhưng sẽ xét lại nếu 1 trong 2 là "nút nhận" trên đường đi tới sink

                # Kiểm tra nếu node2 nằm trên đường từ node1 đến sink => node2 là "nút nhận"
                if not node1.is_sink:
                    if graph_chs.is_on_path_to_sink(id1, id2, sink_node.id):
                        node2.energy -= self.K2 * self.K  # Nhận => tốn ít
                        energy_loss += self.K2 * self.K
                        chs_loss += self.K2 * self.K
                    else:
                        node1.energy -= self.K1 * d * self.K
                        energy_loss += self.K1 * d * self.K
                        chs_loss += self.K1 * d * self.K

                # Kiểm tra nếu node1 nằm trên đường từ node2 đến sink => node1 là "nút nhận"
                if not node2.is_sink:
                    if graph_chs.is_on_path_to_sink(id2, id1, sink_node.id):
                        node1.energy -= self.K2 * self.K  # Nhận => tốn ít
                        energy_loss += self.K2 * self.K
                        chs_loss += self.K2 * self.K
                    else:
                        node2.energy -= self.K1 * d * self.K
                        energy_loss += self.K1 * d * self.K
                        chs_loss += self.K1 * d * self.K
            
            print("chs_loss = ",chs_loss)
            for cluster in self.cluster_infos: 
                ch_id = cluster[0]
                cluster_head = Node.nodes[ch_id]
                for cm_id in cluster[1:]:
                    cluster_member = Node.nodes[cm_id]
                    if not cluster_member.is_sink:
                        distance = cluster_member.distance_to(cluster_head) 
                        cluster_member.energy -= distance * self.K1 * self.K
                        cluster_head.energy -= self.K2 * self.K
                        energy_loss += distance * self.K1 * self.K + self.K2 * self.K
            
        
        if non_accept == 0:
            if self.is_k_connect(self.K):
                self.time_k_connect += 1
                is_k_connect = True
                final_chs = self.cluster_heads
        non_accept = 0
        if(display):
            self.display_network(folder)
            self.display_network(folder="this")
        for node in self.available_nodes:
            node.update_all()
        self.update_network()
        # self.save_network()
        for node in self.available_nodes:
            node.is_cluster_head = False
            node.degree = 1
            node.cluster_index = None  # Đặt lại cluster_index
            node.update_status()
        self.time_life = self.time_life + 1
        # print("length = ", len(self.available_nodes))
        # self.reset()
        # print("chs = ",[node.id for node in final_chs])
        return energy_loss, is_k_connect, final_chs
    
    def display_nodes(self):
        """In ra danh sách các node"""
        for i, node in enumerate(self.available_nodes):
            sink_status = " (Sink Node)" if getattr(node, 'is_sink', False) else ""
            print(f"Node {i}: (x={node.x:.2f}, y={node.y:.2f}){sink_status}")
    
    def display_edges(self):
        """In ra danh sách các cạnh trong mạng"""
        print("Edges:")
        for source, target in self.edges:
            print(f"(Node {self.available_nodes.index(source)}, Node {self.available_nodes.index(target)})")
    
    def take_top(self, M):
        """Chọn M nodes có energy cao nhất từ available_nodes, không chọn sink."""
        return sorted(
            [node for node in self.available_nodes if not node.is_sink], 
            key=lambda node: node.energy, 
            reverse=True
        )[:M]


    def display_network(self, folder):
        """ Hiển thị toàn bộ WSN """
        plt.figure(figsize=(10, 10))
        for node in self.nodes:
            if node.is_sink:
                plt.scatter(node.x, node.y, color=node.color, s=150, marker="^", edgecolors="black")  # Triangle for sink node
            else:
                plt.scatter(node.x, node.y, color=node.color, s=50, marker="o", edgecolors="black")  # Circle for normal nodes
            
            # Thêm hiển thị năng lượng node gần vị trí node
        
        for node in self.available_nodes:
            plt.text(node.x + 1, node.y + 1, f"{node.energy:.1f}", fontsize=7, color='blue', ha='left', va='bottom')

        for edge in self.edges:
            source_id, target_id = edge
            source_node = Node.nodes[source_id]
            target_node = Node.nodes[target_id]
            color = self.edge_colors.get((source_id, target_id), 'black')  # Get color or default to black
            plt.plot([source_node.x, target_node.x], [source_node.y, target_node.y], color=color, alpha=1.0)  # Draw edges

            # Nếu màu là 'purple', in số lên cạnh
            if color == 'purple':
                mid_x = (source_node.x + target_node.x) / 2
                mid_y = (source_node.y + target_node.y) / 2
                edge_value = int(source_node.distance_to(target_node)) # Lấy giá trị trọng số nếu có

                plt.text(mid_x, mid_y, str(edge_value), color='purple', fontsize=7, ha='center', va='center')
        

         # Hiển thị số vòng đời hiện tại
        plt.text(0.05, 1.01, f"Life time: {self.time_life}", transform=plt.gca().transAxes, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
        plt.text(0.05, 1.05, f"Available nodes: {len(self.available_nodes)}", transform=plt.gca().transAxes, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
        plt.text(0.4, 1.05, f"Error: {self.error}", transform=plt.gca().transAxes, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Wireless Sensor Network Visualization")
        plt.grid(True)

        # Lưu ảnh vào thư mục image với tên theo thời gian sống
        filename = f"{folder}/network_life_{self.time_life}.png"
        plt.savefig(filename, bbox_inches='tight')
        print("save in ", filename)
        plt.close()

    def update_network(self):
        """Cập nhật mạng: loại bỏ các node chết và giảm số lượng node"""
        self.available_nodes = [node for node in self.available_nodes if not node.is_dead]  # Giữ lại node còn sống
        self.num_nodes = len(self.available_nodes)  # Cập nhật số lượng node
        if(self.num_nodes == 0):
            self.is_dead = True
        self.num_nodes_history.append(self.num_nodes)

    def save_network(self, filename="network"):
        """Lưu toàn bộ thông tin của mạng vào một file JSON với timestamp trong tên file"""

        data = {
            "width": self.width,
            "height": self.height,
            "num_nodes_history": self.num_nodes_history,
            "sink": {
                "x": self.sink_node.x,
                "y": self.sink_node.y
            },
            "nodes": [
                {
                    "id": node.id,
                    "x": node.x,
                    "y": node.y,
                    "is_cluster_head": node.is_cluster_head_history,
                    "energy": node.energy_history,
                    "is_dead": node.life_history,
                    "cluster_index": node.cluster_index_history
                }
                for node in self.nodes
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        # print(f"Network saved to {filename}")
