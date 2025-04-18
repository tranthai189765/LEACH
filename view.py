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

# Đọc dữ liệu từ hai file
fmc_timesteps, fmc_nodes = read_data("FMC_log.txt")
leach_timesteps, leach_nodes = read_data("LEACH_log.txt")

# Vẽ biểu đồ so sánh
plt.figure(figsize=(10, 5))
plt.plot(fmc_timesteps, fmc_nodes, label='FMC', color='red', markersize=2, marker='o')
plt.plot(leach_timesteps, leach_nodes, label='LEACH', color='blue', markersize=2,  marker='s')

# Định dạng biểu đồ
plt.xlabel("Timestep")
plt.ylabel("Number of available nodes")
plt.title("Comparison of FMC and LEACH over time")
plt.legend()
plt.grid()

# Hiển thị biểu đồ
plt.show()
