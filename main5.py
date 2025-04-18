from network import Network
from FCM import run

WSN = Network(num_nodes=1000, seed=45)
WSN.save_network()
WSN.display_network()
for i in range(1000):
    run(WSN, P=0.1, K1=0.9, K2=0.9, K=2, display=True)
    print("Timestep: ", i, " - Number of available nodes: ", len(WSN.available_nodes), "- Time K-connect: ", WSN.time_k_connect)
    if(len(WSN.available_nodes) == 0):
        print("Time K-connect: ", WSN.time_k_connect)
        print("Time life: ", WSN.time_life)
        break

