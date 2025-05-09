import config
from src.comms.ipc_comm import IPCFederatedCommunicator
from src.comms.mpi_comm import MPIFederatedCommunicator
from src.comms.sam_comm import SAMFederatedCommunicator

""" 
Federated Average Algorithm https://arxiv.org/pdf/1602.05629 Algorithm 1
The FedAVG algorithm is a method for aggregating the weights and biases of multiple models trained on different clients.
Essentially combining the weights and biases of the models by taking the average of the weights
and biases weighted by the number of samples each model was trained on.
This implementation differs a bit from the referenced paper.
Although the logic is the same, this implementation also loops through the layers of the model
and calculates the average of the weights and biases for each layer separately. See the function update_model in comm.py
"""
def main():
    communicator = None

    if config.COMM == "mpi": communicator = MPIFederatedCommunicator()
    elif config.COMM == "ipc": communicator = IPCFederatedCommunicator()
    elif config.COMM == "sam": communicator = SAMFederatedCommunicator()

    communicator.run()

if __name__ == "__main__":
    main()