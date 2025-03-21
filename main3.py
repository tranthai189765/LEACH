from network import Network
from LEACH_basic import run

WSN = Network(100)
WSN.save_network()
WSN.display_network()

for i in range(1000):
    run(WSN, 10/len(WSN.available_nodes), 0.1, 0.2,display=True)

