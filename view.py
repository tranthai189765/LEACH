import matplotlib.pyplot as plt
import re

def read_data(file_path):
    """ Đọc file log và trích xuất dữ liệu timeStep và số node. """
    timesteps = []
    node_counts = []
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'Timestep:\s+(\d+)\s+-\s+Number of available nodes:\s+(\d+)', line)
            if match:
                timestep = int(match.group(1))
                node_count = int(match.group(2))
                timesteps.append(timestep)
                node_counts.append(node_count)
    
    return timesteps, node_counts

# Danh sách các file và nhãn tương ứng
files = [
    ("FMC_log.txt", "FMC", 'red', 'o'),
    ("LEACH.txt", "LEACH", 'blue', 's'),
    ("I_LEACH1.txt", "LEACH_2018", 'green', '^'),
    ("I_LEACH2.txt", "LEACH_2016", 'purple', 'd')
]

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))

for file_path, label, color, marker in files:
    timesteps, node_counts = read_data(file_path)
    plt.plot(timesteps, node_counts, label=label, color=color, marker=marker, markersize=2)

# Định dạng biểu đồ
plt.xlabel("Timestep")
plt.ylabel("Number of available nodes")
plt.title("Comparison of Clustering Protocols Over Time")
plt.legend()
plt.grid()

# Hiển thị biểu đồ
plt.show()
