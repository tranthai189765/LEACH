import sys
from network import Network
from FCM import run 
import time
from LEACH_basic_updated import run as run_1
from LEACH_2018 import run as run_2

# Mở file để ghi output
with open("output_log200100.txt", "w", buffering=1) as f:
    sys.stdout = f  # Chuyển hướng toàn bộ output của print() vào file

    WSN = Network(num_nodes=100, seed=64)
    # WSN.display_network()
    WSN.save_network()
    # WSN.display_network()
    for i in range(1000):
        run(WSN, P=0.1, K1=0.1, K2=0.1, K=2, display=False)
        print("Timestep: ", i, " - Number of available nodes: ", len(WSN.available_nodes), "- Time K-connect: ", WSN.time_k_connect)
        if (len(WSN.available_nodes) == 0) or (WSN.is_dead is True):
            print("Time K-connect: ", WSN.time_k_connect)
            print("Time life: ", WSN.time_life)
            break
    print("done 1 cai roi")

    WSN = Network(num_nodes=100, seed=64)
    for i in range(1000):
        run_1(WSN, P=0.1, K1=0.1, K2=0.1, K=2, display=False)
        print("Timestep: ", i, " - Number of available nodes: ", len(WSN.available_nodes), "- Time K-connect: ", WSN.time_k_connect)
        if (len(WSN.available_nodes) == 0) or (WSN.is_dead is True):
            print("Time K-connect: ", WSN.time_k_connect)
            print("Time life: ", WSN.time_life)
            break
    
    print("done 2 cai roi")
    WSN = Network(num_nodes=100, seed=64)
    WSN.save_network()
    for i in range(1000):
        if len(WSN.available_nodes) > 100:
            run_1(WSN, P=100/len(WSN.available_nodes), K1=0.2, K2=0.02, K=2, display=False)
        else:
            run_1(WSN, P=0.1, K1=0.2, K2=0.02, K=2, display=False)
        current = 0
        print("Timestep: ", i, " - Number of available nodes: ", len(WSN.available_nodes), "- Time K-connect: ", WSN.time_k_connect)
        if(len(WSN.available_nodes) == 0):
            print("Time K-connect: ", WSN.time_k_connect)
            print("Time life: ", WSN.time_life)
            break
    
    print("done 3 cai roi")
    WSN = Network(num_nodes=100, seed=64)
    for i in range(1000):
        run_2(WSN, P=0.1, K1=0.1, K2=0.1, K=2, display=False)
        print("Timestep: ", i, " - Number of available nodes: ", len(WSN.available_nodes), "- Time K-connect: ", WSN.time_k_connect)
        if (len(WSN.available_nodes) == 0) or (WSN.is_dead is True):
            print("Time K-connect: ", WSN.time_k_connect)
            print("Time life: ", WSN.time_life)
            break

    sys.stdout = sys.__stdout__  # Khôi phục stdout về bình thường
# Mở file để ghi output
with open("output_log30.txt", "w", buffering=1) as f:
    sys.stdout = f  # Chuyển hướng toàn bộ output của print() vào file

    WSN = Network(num_nodes=1000, seed=49)
    WSN.save_network()
    # WSN.display_network()
    for i in range(1000):
        run(WSN, P=0.1, K1=0.1, K2=0.1, K=2, display=False)
        print("Timestep: ", i, " - Number of available nodes: ", len(WSN.available_nodes), "- Time K-connect: ", WSN.time_k_connect)
        if (len(WSN.available_nodes) == 0) or (WSN.is_dead is True):
            print("Time K-connect: ", WSN.time_k_connect)
            print("Time life: ", WSN.time_life)
            break

    WSN = Network(num_nodes=1000, seed=49)
    for i in range(1000):
        run_1(WSN, P=0.1, K1=0.1, K2=0.1, K=2, display=False)
        print("Timestep: ", i, " - Number of available nodes: ", len(WSN.available_nodes), "- Time K-connect: ", WSN.time_k_connect)
        if (len(WSN.available_nodes) == 0) or (WSN.is_dead is True):
            print("Time K-connect: ", WSN.time_k_connect)
            print("Time life: ", WSN.time_life)
            break

    sys.stdout = sys.__stdout__  # Khôi phục stdout về bình thường