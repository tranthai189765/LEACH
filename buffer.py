class ClusterHeadBuffer:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.buffer = [None] * mem_size  # Khởi tạo danh sách rỗng với kích thước cố định

    def store(self, list_cluster_heads):
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
buffer = ClusterHeadBuffer(mem_size=5)

buffer.store([1, 2])
'''
buffer.store([4, 5])
buffer.store([3, 6])
buffer.store([7])
buffer.store([8, 9])
buffer.store([10])  # Ghi đè vào vị trí đầu tiên do buffer đầy
buffer.store([11,12,13])'
'''
print(buffer.buffer)

print(buffer.take(3))  # Output: {3, 6, 7, 8, 9, 10}