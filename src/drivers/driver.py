import config
from src.data_loader import load_dataset, partition_training_data
from src.comms.ipc_comm import IPCFederatedCommunicator
from src.comms.mpi_comm import MPIFederatedCommunicator

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
    (train_imgs, train_lbls), test_set = load_dataset()
    partitions = partition_training_data(train_imgs, train_lbls, config.N_PARTITIONS)
    
    data_sizes = [len(p) for p in partitions]
    communicator = None

    if config.COMM == "mpi": communicator = MPIFederatedCommunicator()
    elif config.COMM == "ipc": communicator = IPCFederatedCommunicator()

    global_model = communicator.init_global_model()

    for epoch in range(config.N_EPOCHS):
        data_stack = communicator.create_data_stack(global_model, partitions)
        communicator.distribute_data(data_stack)
        local_models = communicator.collect_models()
        global_model = communicator.update_model(local_models, data_sizes, test_set, epoch)

if __name__ == "__main__":
    main()            