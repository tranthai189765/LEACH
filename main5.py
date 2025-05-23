from network import Network
from LEACH_2020 import run

WSN = Network(num_nodes=100, seed=64)
WSN.save_network()
for i in range(1000):
    print("before : ", len(WSN.available_nodes))
    run(WSN, P=0.1, K1=100.0, K2=100.0, K=2, display=False)
    print("Timestep: ", i, " - Number of available nodes: ", len(WSN.available_nodes), "- Time K-connect: ", WSN.time_k_connect)
    if(len(WSN.available_nodes) == 0):
        print("Time K-connect: ", WSN.time_k_connect)
        print("Time life: ", WSN.time_life)
        break

