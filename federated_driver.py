#!/usr/bin/env python3
# Federated MNIST Trainer Using Python Subprocess
# Based on Federated Learning https://arxiv.org/pdf/1602.05629   

import subprocess
import pickle
import os
import mnist
import config
from data_loader import load_dataset, create_partition_files, create_test_file

def FedAVG(models, data_sizes):
    total_size = sum(data_sizes)
    combined_biases = []
    combined_weights = []
    for layer in range(len(models[0].biases)): # should be a constant in config, but for now we will assume all models have the same number of layers
        wb = sum((model.biases[layer] * data_sizes[i] for i, model in enumerate(models)) / total_size)
        combined_biases.append(wb)
        ww = sum((model.weights[layer] * data_sizes[i] for i, model in enumerate(models)) / total_size)
        combined_weights.append(ww)
    return combined_biases, combined_weights

def spawn_workers(partition_files, test_file, init_file):
    processes = []
    out_files = []
    for i, pfile in enumerate(partition_files):
        out_fname = f"trained_model_{i}.pkl"
        out_files.append(out_fname)
        cmd = ["python3", "-u", "worker.py", "--partition", pfile, "--test", test_file, "--output", out_fname]
        if init_file:
            cmd.extend(["--initial", init_file])
        if config.DEBUG:
            print("Spawning process:", " ".join(cmd))
            proc = subprocess.Popen(cmd)
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((proc, i, out_fname))
    return processes, out_files

def wait_for_workers(processes):
    for proc, i, _ in processes:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0 and config.DEBUG:
            print(f"Error in worker for partition {i}:")
            print("STDOUT:", stdout.decode("utf-8"))
            print("STDERR:", stderr.decode("utf-8"))
        elif proc.returncode == 0 and config.DEBUG:
            print(f"Worker for partition {i} completed successfully.")

def load_worker_outputs(out_files):
    models = []
    for fname in out_files:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                models.append(pickle.load(f))
    return models

def cleanup(files):
    for f in files:
        if os.path.exists(f):
            os.remove(f)

# Federated Average Algorithm https://arxiv.org/pdf/1602.05629 Algorithm 1
# The FedAVG algorithm is a method for aggregating the weights and biases of multiple models trained on different clients.
# Essentially combining the weights and biases of the models by taking the average of the weights
# and biases weighted by the number of samples each model was trained on.
def main():
    (train_imgs, train_lbls), (test_imgs, test_lbls) = load_dataset()

    # St <-- (random set of m clients) NOTE: I take the whole set not random.
    # In FedAVG, m is the number of clients selected in each round.
    # Here, we partition the data into config.N_PARTITIONS clients and run them every round,
    # so effectively, m = config.N_PARTITIONS (i.e. all clients participate each round).
    partition_files, data_sizes = create_partition_files(train_imgs, train_lbls, config.N_PARTITIONS)
    test_file = create_test_file(test_imgs, test_lbls, config.TEST_SAMPLE_SIZE)

    # Initialize W0 
    final_model = None

    # For each round t = 1,2,... do 
    for epoch in range(config.N_EPOCHS):
        print(f"\n=== Training Epoch {epoch} ===")

        # (Load Previous Model if available)
        init_file = None
        if final_model:
            init_file = "initial_model.pkl"
            with open(init_file, "wb") as f:
                pickle.dump(final_model, f)

        # (Spawn Workers)
        # For each client k in St in parallel do
        # Wkt+1 <-- ClientUpdate(k, Wt)         
        procs, out_files = spawn_workers(partition_files, test_file, init_file)
        wait_for_workers(procs)
        print(out_files)
        models = load_worker_outputs(out_files)
        
        # (Aggregate Models)
        # Wt+1 <-- FedAVG(St, wkt+1)
        biases, weights = FedAVG(models, data_sizes)
        final_model = mnist.Network(config.NETWORK_ARCHITECTURE)
        final_model.biases = biases
        final_model.weights = weights

        # (Evaluate Final Model)
        full_test = list(zip(test_imgs, test_lbls))
        print(f"Epoch {epoch} Final Model Evaluation: {final_model.evaluate(full_test)} / {len(full_test)}")
        
        # (Cleanup temp files)
        cleanup(out_files)
        if init_file:
            cleanup([init_file])
    cleanup(partition_files + [test_file])

if __name__ == "__main__":
    main()
