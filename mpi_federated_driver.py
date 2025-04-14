#!/usr/bin/env python3
# Federated MNIST Trainer using MPI with Refactored Epoch Loop
# Usage: mpiexec -n <config.N_PARTITIONS+1> python3 mpi_federated.py
# Rank 0 is the master; ranks 1..N_PARTITIONS are workers.
# Based on Federated Averaging (FedAVG) from https://arxiv.org/pdf/1602.05629.

from mpi4py import MPI  # type: ignore
import mnist
import config
from data_loader import load_dataset, partition_training_data

def FedAVG(models, data_sizes):
    total_size = sum(data_sizes)
    combined_biases = []
    combined_weights = []
    for layer in range(len(models[0].biases)):
        weighted_bias = sum(model.biases[layer] * data_sizes[i]
                            for i, model in enumerate(models)) / total_size
        weighted_weight = sum(model.weights[layer] * data_sizes[i]
                              for i, model in enumerate(models)) / total_size
        combined_biases.append(weighted_bias)
        combined_weights.append(weighted_weight)
    return combined_biases, combined_weights

def train_local_model(global_model, local_data, test_data):
    if global_model is not None:
        net = mnist.Network(global_model.sizes)
        net.biases = [b.copy() for b in global_model.biases]
        net.weights = [w.copy() for w in global_model.weights]
    else:
        net = mnist.Network(config.NETWORK_ARCHITECTURE)
    net.SGD(local_data, epochs=config.SGD_EPOCHS, mini_batch_size=config.MINI_BATCH_SIZE,
            eta=config.ETA, test_data=test_data)
    return net

def master_epoch(epoch, partitions, data_sizes, test_data, global_model, comm):
    
    print(f"\n=== Training Epoch {epoch} ===")

    # Broadcast current global model and test data.
    global_model = comm.bcast(global_model, root=0)
    test_data = comm.bcast(test_data, root=0)

    # Send each worker its local partition.
    for i in range(config.N_PARTITIONS):
        comm.send(partitions[i], dest=i+1, tag=11)

    # Collect updated models from workers.
    local_models = []
    for i in range(1, config.N_PARTITIONS+1):
        local_model = comm.recv(source=i, tag=22)
        local_models.append(local_model)

    # Aggregate local models using FedAVG.
    # Wt+1 <-- FedAVG(St, wkt+1)
    biases, weights = FedAVG(local_models, data_sizes)
    new_model = mnist.Network(config.NETWORK_ARCHITECTURE)
    new_model.biases = biases
    new_model.weights = weights

    # Evaluate 
    full_test = list(zip(*load_dataset()[1]))
    print(f"Epoch {epoch} Final Model Evaluation: {new_model.evaluate(full_test)} / {len(full_test)}")
    return new_model

def worker_epoch(epoch, comm):
    # Each worker receives the broadcasted global model and test data.
    global_model = comm.bcast(None, root=0)
    test_data = comm.bcast(None, root=0)
    if config.DEBUG: print(f"[DEBUG] Worker {comm.Get_rank()}: Epoch {epoch}: Received global model and test data.", flush=True)
    local_data = comm.recv(source=0, tag=11)
    if config.DEBUG: print(f"[DEBUG] Worker {comm.Get_rank()}: Epoch {epoch}: Received local data for training.", flush=True)
    local_model = train_local_model(global_model, local_data, test_data)
    if config.DEBUG: print(f"[DEBUG] Worker {comm.Get_rank()}: Epoch {epoch}: Completed local training.", flush=True)
    comm.send(local_model, dest=0, tag=22)
    if config.DEBUG: print(f"[DEBUG] Worker {comm.Get_rank()}: Epoch {epoch}: Sent local model to master.", flush=True)

# Federated Average Algorithm https://arxiv.org/pdf/1602.05629 Algorithm 1
# The FedAVG algorithm is a method for aggregating the weights and biases of multiple models trained on different clients.
# Essentially combining the weights and biases of the models by taking the average of the weights
# and biases weighted by the number of samples each model was trained on.
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    required_size = config.N_PARTITIONS + 1
    if size < required_size:
        if rank == 0:
            print(f"Error: Requires at least {required_size} processes (1 master + {config.N_PARTITIONS} workers).")
        exit(1)

    n_epochs = config.N_EPOCHS   
    if rank == 0:
        # MASTER: Load dataset and partition data.
        (train_imgs, train_lbls), (test_imgs, test_lbls) = load_dataset()

        # St <-- (random set of m clients) NOTE: I take the whole set not random.
        # In FedAVG, m is the number of clients selected in each round.
        # Here, we partition the data into config.N_PARTITIONS clients and run them every round,
        # so effectively, m = config.N_PARTITIONS (i.e. all clients participate each round).
        partitions = partition_training_data(train_imgs, train_lbls, config.N_PARTITIONS)
        data_sizes = [len(part) for part in partitions]
        test_data = list(zip(test_imgs, test_lbls))[:config.TEST_SAMPLE_SIZE]

        # Initialize W0 
        global_model = None

        # For each client k in St in parallel do
        # Wkt+1 <-- ClientUpdate(k, Wt)
        for epoch in range(n_epochs):
            global_model = master_epoch(epoch, partitions, data_sizes, test_data, global_model, comm)
        # Optionally, broadcast a termination signal.
        comm.bcast(None, root=0)
    else:
        # WORKERS: Loop through epochs.

        # For each client k in St in parallel do
        # Wkt+1 <-- ClientUpdate(k, Wt)
        for epoch in range(n_epochs):
            worker_epoch(epoch, comm)

        # Final broadcast to signal termination.
        comm.bcast(None, root=0)

if __name__ == "__main__":
    main()
