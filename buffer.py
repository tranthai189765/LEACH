class ClusterHeadBufferTest:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.buffer = [None] * mem_size  # Khởi tạo danh sách rỗng với kích thước cố định

    def add(self, list_cluster_heads):
        index = self.mem_cntr % self.mem_size
        self.buffer[index] = list_cluster_heads
        self.mem_cntr += 1

    def take(self, num_steps):
        collected_nodes = set()
        for i in range(1, num_steps + 1):
            index = (self.mem_cntr - i) % self.mem_size
            if self.mem_cntr - i < 0 or self.buffer[index] is None:
                break  # Dừng nếu không còn phần tử hợp lệ
            collected_nodes.update(self.buffer[index])
        return collected_nodes