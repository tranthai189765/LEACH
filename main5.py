from network import Network
from A_LEACH import run

WSN = Network(100)
WSN.save_network()
WSN.display_network()

for i in range(1000):
    run(WSN, 5, 0.1, 0.1, 0.2,display=True)
