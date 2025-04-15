import pickle
import config
from data_loader import load_dataset, create_partition_files, create_test_file
from ipc_comm import IPCFederatedCommunicator

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
    partition_files, data_sizes = create_partition_files(train_imgs, train_lbls, config.N_PARTITIONS)
    test_file = create_test_file(test_imgs, test_lbls, config.TEST_SAMPLE_SIZE)
    communicator = IPCFederatedCommunicator(partition_files, test_file)
    global_model = None

    for epoch in range(config.N_EPOCHS):
        print(f"\n=== Training Epoch {epoch} ===")

        init_file = None
        if global_model:
            init_file = "temp/initial_model.pkl"
            with open(init_file, "wb") as f:
                pickle.dump(global_model, f)

        processes, out_files = communicator.distribute_data(init_file)
        communicator.wait_for_completion(processes)
        models = communicator.collect_models(out_files)
        global_model = communicator.update_model(global_model, models, data_sizes)

        full_test = list(zip(test_imgs, test_lbls))
        print(f"Epoch {epoch} Final Model Evaluation: {global_model.evaluate(full_test)} / {len(full_test)}")

        communicator.cleanup(out_files)
        if init_file:
            communicator.cleanup([init_file])

    communicator.cleanup(partition_files + [test_file])

if __name__ == "__main__":
    main()
