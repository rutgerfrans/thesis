from mpi4py import MPI #type: ignore
import config
from data_loader import load_dataset, partition_training_data
from mpi_comm import MPIFederatedCommunicator

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
    required_size = config.N_PARTITIONS + 1
    if size < required_size:
        if rank == 0:
            print(f"Error: Requires at least {required_size} processes (1 master + {config.N_PARTITIONS} workers).")
        exit(1)

    communicator = MPIFederatedCommunicator(comm)
    n_epochs = config.N_EPOCHS

    if rank == 0:
        # MASTER
        (train_imgs, train_lbls), (test_imgs, test_lbls) = load_dataset()
        partitions = partition_training_data(train_imgs, train_lbls, config.N_PARTITIONS)
        data_sizes = [len(p) for p in partitions]
        test_data = list(zip(test_imgs, test_lbls))[:config.TEST_SAMPLE_SIZE]
        global_model = None

        for epoch in range(n_epochs):
            print(f"\n=== Training Epoch {epoch} ===")

            communicator.broadcast_model(global_model, test_data)
            communicator.distribute_data(partitions)

            local_models = communicator.collect_models(config.N_PARTITIONS)
            global_model = communicator.update_model(global_model, local_models, data_sizes)

            full_test = list(zip(test_imgs, test_lbls))
            print(f"Epoch {epoch} Final Model Evaluation: {global_model.evaluate(full_test)} / {len(full_test)}")

        comm.bcast(None, root=0)  # Signal termination

    else:
        # WORKER
        for epoch in range(n_epochs):
            communicator.worker_epoch(epoch)
        comm.bcast(None, root=0)  # Receive termination

if __name__ == "__main__":
    main()
