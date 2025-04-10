#!/usr/bin/env python3
import pickle
import argparse
import mnist
import config

def load_partition(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_test_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_initial_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def train_partition(partition, test_data, initial_model=None):
    if initial_model is not None:
        net = mnist.Network(initial_model.sizes)
        net.biases = [b.copy() for b in initial_model.biases]
        net.weights = [w.copy() for w in initial_model.weights]
    else:
        net = mnist.Network(config.NETWORK_ARCHITECTURE)
    print("Initial evaluation on partition:", net.evaluate(test_data), "/", len(test_data), flush=True)
    net.SGD(partition, epochs=config.SGD_EPOCHS, mini_batch_size=config.MINI_BATCH_SIZE, eta=config.ETA, test_data=test_data)
    print("Post-training evaluation:", net.evaluate(test_data), "/", len(test_data), flush=True)
    return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial", default=None)
    args = parser.parse_args()
    partition = load_partition(args.partition)
    test_data = load_test_data(args.test)
    initial_model = load_initial_model(args.initial) if args.initial else None
    trained_model = train_partition(partition, test_data, initial_model)
    with open(args.output, "wb") as f:
        pickle.dump(trained_model, f)

if __name__ == "__main__":
    main()
