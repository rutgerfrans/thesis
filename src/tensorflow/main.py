#!/usr/bin/env python3
import os
import glob
import json
import random
import time
import csv
from datetime import datetime

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ["TF_CPP_VLOG_LEVEL"] = "99"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import tensorflow as tf

import config
import src.mnist as mnist
from src.data_loader import load_dataset

fault_p = config.FAULT_P

def clear_snapshots(ckpt_dir):
    for fname in glob.glob(os.path.join(ckpt_dir, 'ckpt-*')):
        try:
            os.remove(fname)
            print(f"Deleted checkpoint: {fname}")
        except OSError as e:
            print(f"Error deleting {fname}: {e}")
    try:
        os.remove(os.path.join(ckpt_dir, 'checkpoint'))
    except OSError:
        pass

def main():
    if 'TF_CONFIG' not in os.environ:
        default = {
            'cluster': {'worker': ['localhost:12345']},
            'task':    {'type': 'worker', 'index': 0}
        }
        os.environ['TF_CONFIG'] = json.dumps(default)

    resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    task_type, task_id = resolver.task_type, resolver.task_id
    is_chief = (task_type == 'worker' and task_id == 0)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    num_workers = strategy.num_replicas_in_sync

    partitions, test_data = load_dataset()
    local_part = partitions[task_id]

    training_data = []
    for img, label in local_part:
        x = img.astype(np.float32).reshape(-1, 1)
        y = label.astype(np.float32).reshape(-1, 1)

        training_data.append((x, y))

    ckpt_dir = os.path.join(os.getcwd(), 'src', 'tensorflow', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    with strategy.scope():
        network = mnist.Network(config.NETWORK_ARCHITECTURE)
        tf_weights = [tf.Variable(w.astype(np.float32), trainable=False) for w in network.weights]
        tf_biases  = [tf.Variable(b.astype(np.float32), trainable=False) for b in network.biases]

        ckpt = tf.train.Checkpoint(
            **{f"w{i}": v for i, v in enumerate(tf_weights)},
            **{f"b{j}": v for j, v in enumerate(tf_biases)}
        )
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=None)

        latest = manager.latest_checkpoint
        start_epoch = 0
        if latest:
            ckpt.restore(latest).expect_partial()
            fn = os.path.basename(latest)
            try:
                start_epoch = int(fn.split('-')[-1])
            except ValueError:
                start_epoch = 0
            if is_chief:
                print(f"Restored checkpoint {fn}, starting at epoch {start_epoch}")

    compute_time_total = 0.0
    timing_csv = os.path.join(os.getcwd(), "src", "tensorflow", "timings", "epoch_timings.csv")
    if not os.path.exists(timing_csv):
        os.makedirs(os.path.dirname(timing_csv), exist_ok=True)
        with open(timing_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["system_overhead", "avg_compute_time_per_worker"])

    run_start = datetime.now()
    if is_chief:
        print(f"Starting training at {run_start.isoformat()}, workers={num_workers}")

    for epoch in range(start_epoch, config.N_EPOCHS):
        if fault_p > 0.0 and random.random() < fault_p:
            print(f"[worker {task_id}] Injecting fault at epoch {epoch}")
            os._exit(1)

        t0 = time.perf_counter()

        for i, v in enumerate(tf_weights):
            network.weights[i] = v.numpy()
        for j, v in enumerate(tf_biases):
            network.biases[j] = v.numpy()

        network.SGD(
            training_data=training_data,
            epochs=config.SGD_EPOCHS,
            mini_batch_size=config.MINI_BATCH_SIZE,
            eta=config.ETA
        )

        for i, v in enumerate(tf_weights):
            v.assign(network.weights[i])
        for j, v in enumerate(tf_biases):
            v.assign(network.biases[j])

        for v in tf_weights + tf_biases:
            reduced = tf.raw_ops.CollectiveReduce(
                input=v.read_value(),
                group_size=num_workers,
                group_key=1,
                instance_key=epoch + 1,
                merge_op='Add',
                final_op='Id',
                subdiv_offsets=[0]
            )
            v.assign(reduced / tf.cast(num_workers, reduced.dtype))

        t1 = time.perf_counter()
        epoch_compute = t1 - t0
        compute_time_total += epoch_compute

        if is_chief:
            print(f"[Epoch {epoch+1}] compute_time={epoch_compute:.4f}s")
            manager.save(checkpoint_number=epoch+1)

    if is_chief:
        clear_snapshots(ckpt_dir)
        run_end = datetime.now()
        total_wall = (run_end - run_start).total_seconds()
        system_overhead = total_wall - (compute_time_total / num_workers)
        print(
            f"Training done at {run_end.isoformat()}; "
            f"total_compute={compute_time_total:.4f}s; "
            f"wall_time={total_wall:.4f}s; "
            f"system_overhead={system_overhead:.4f}s"
        )
        with open(timing_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{system_overhead:.6f}", f"{(compute_time_total/num_workers):.6f}"])

        correct = 0
        total   = 0

        for img, label in test_data:
            x_test = img.astype(np.float32).reshape(-1, 1)
            output = network.feedforward(x_test)
            predicted_label = np.argmax(output)
            true_label = int(label)

            if predicted_label == true_label:
                correct += 1
            total += 1

        if total > 0:
            accuracy = correct / total
            print(f"Test accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
        else:
            print("Warning: test_data is empty; cannot compute accuracy.")

    os._exit(0)

if __name__ == '__main__':
    main()
