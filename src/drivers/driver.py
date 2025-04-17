import config
from mpi4py import MPI #type: ignore
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
    (train_imgs, train_lbls), (test_imgs, test_lbls) = load_dataset()
    partitions = partition_training_data(train_imgs, train_lbls, config.N_PARTITIONS)
    data_sizes = [len(p) for p in partitions]
    global_model = None

    if config.COMM == "mpi":
        comm = MPI.COMM_WORLD
        communicator = MPIFederatedCommunicator(comm)

        data_stack = [[part, global_model] for part in partitions]

        for epoch in range(config.N_EPOCHS):
            data = communicator.distribute_data(data_stack) # only worker 0 should do this but scatter does not work like this 
            local_model = communicator.train_model(data[1], data[0])
            local_models = communicator.collect_models(local_model) # only worker 0 should do this but gather doenst work like this

            if comm.Get_rank() == 0:            
                global_model = communicator.update_model(global_model, local_models, data_sizes)
                eval = communicator.evaluate_model(global_model, list(zip(test_imgs, test_lbls)), epoch)
                print(eval)

            data_stack = [(part, global_model) for part in partitions]

    if config.COMM == "ipc":
        communicator = IPCFederatedCommunicator()
        
        partition_files = communicator.create_partition_files(partitions)

        for epoch in range(config.N_EPOCHS):
            processes, local_model_files = communicator.distribute_data(global_model, partition_files)
            communicator.wait_for_completion(processes)

            models = communicator.collect_models(local_model_files)
            global_model = communicator.update_model(global_model, models, data_sizes)

            eval = communicator.evaluate_model(global_model, list(zip(test_imgs, test_lbls)), epoch)
            print(eval)

            communicator.cleanup(local_model_files + [communicator.global_model_file])

        communicator.cleanup(partition_files)

if __name__ == "__main__":
    main()            