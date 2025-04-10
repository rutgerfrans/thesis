#!/usr/bin/env python3
import pickle
import numpy as np

def load_images(filename):
    images = []
    with open(filename, "rb") as f:
        raw = f.read()
        framesize = 28 * 28
        headersize = 4 * 4
        raw = raw[headersize:]
        for i in range(0, len(raw) // framesize):
            pixels = np.array([v / 255.0 for v in raw[framesize * i: framesize * (i + 1)]])
            images.append(pixels.reshape((28 * 28, 1)))
    return images

def load_labels(filename):
    labels = []
    with open(filename, "rb") as f:
        raw = f.read()
        headersize = 4 * 2
        raw = raw[headersize:]
        for v in raw:
            labels.append(v)
    return labels

def load_dataset():
    train_imgs = load_images('dataset/train-images.idx3-ubyte')
    train_lbls = load_labels('dataset/train-labels.idx1-ubyte')
    test_imgs = load_images('dataset/t10k-images.idx3-ubyte')
    test_lbls = load_labels('dataset/t10k-labels.idx1-ubyte')
    return (train_imgs, train_lbls), (test_imgs, test_lbls)

def partition_training_data(images, labels, n):
    data = list(zip(images, map(lambda v: np.eye(10)[v].reshape((10, 1)), labels)))
    chunk_size = len(data) // n
    return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]

def create_partition_files(train_imgs, train_lbls, n):
    partitions = partition_training_data(train_imgs, train_lbls, n)
    files = []
    sizes = []
    for i, part in enumerate(partitions):
        fname = f"partition_{i}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(part, f)
        files.append(fname)
        sizes.append(len(part))
    return files, sizes

def create_test_file(test_imgs, test_lbls, count):
    test_data = list(zip(test_imgs, test_lbls))[:count]
    fname = "test_data.pkl"
    with open(fname, "wb") as f:
        pickle.dump(test_data, f)
    return fname
