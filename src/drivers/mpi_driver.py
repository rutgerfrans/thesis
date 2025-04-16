from mpi4py import MPI #type: ignore
import config
from src.data_loader import load_dataset, partition_training_data
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    required_size = config.N_PARTITIONS

    if size < required_size and rank == 0: print(f"Error: Requires at least {required_size} processes (1 master + {config.N_PARTITIONS} workers)."), exit(1)

    (train_imgs, train_lbls), (test_imgs, test_lbls) = load_dataset()
    partitions = partition_training_data(train_imgs, train_lbls, config.N_PARTITIONS)
    data_sizes = [len(p) for p in partitions]
    test_data = list(zip(test_imgs, test_lbls))[:config.TEST_SAMPLE_SIZE]
    global_model = None
    data_stack = [[part, global_model] for part in partitions]

    communicator = MPIFederatedCommunicator(comm)

    for epoch in range(config.N_EPOCHS):
        
        partition = comm.scatter(data_stack)
        local_model = communicator.worker(partition, epoch, test_data)
        local_models = comm.gather(local_model, root=0)

        if rank == 0:            
            global_model = communicator.update_model(global_model, local_models, data_sizes)

            full_test = list(zip(test_imgs, test_lbls))
            print(f"Epoch {epoch} Final Model Evaluation: {global_model.evaluate(full_test)} / {len(full_test)}")

        data_stack = [(part, global_model) for part in partitions]

if __name__ == "__main__":
    main()
