from network import Network
from LEACH import run

WSN = Network(100)
WSN.save_network()
WSN.display_network()

for i in range(10):
    run(WSN, 20, 0.1, 0.2,display=True)

