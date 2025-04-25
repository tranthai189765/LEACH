import matplotlib.pyplot as plt
import re

def read_k_connect_data(file_path):
    """ Đọc file log và trích xuất dữ liệu timestep và Time K-connect. """
    timesteps = []
    k_connect_times = []
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'Timestep:\s+(\d+)\s+-\s+Number of available nodes:\s+\d+\s+-\s+Time K-connect:\s+(\d+)', line)
            if match:
                timestep = int(match.group(1))
                time_k_connect = int(match.group(2))
                timesteps.append(timestep)
                k_connect_times.append(time_k_connect)
    
    return timesteps, k_connect_times

# Đọc dữ liệu từ hai file log
fmc_timesteps, fmc_kconnect = read_k_connect_data("FMC_log.txt")
leach_timesteps, leach_kconnect = read_k_connect_data("LEACH_log.txt")

# Vẽ biểu đồ so sánh Time K-connect
plt.figure(figsize=(10, 5))
plt.plot(fmc_timesteps, fmc_kconnect, label='FMC', color='red', marker='o', markersize=2)
plt.plot(leach_timesteps, leach_kconnect, label='LEACH', color='blue', marker='s', markersize=2)

# Định dạng biểu đồ
plt.xlabel("Timestep")
plt.ylabel("Time K-connect")
plt.title("Comparison of Time K-connect between FMC and LEACH")
plt.legend()
plt.grid()

# Hiển thị biểu đồ
plt.show()
