#!/usr/bin/env python3
import os, glob, argparse, time, csv, random
from datetime import datetime
import numpy as np
import torch
from torch.distributed import (init_process_group, destroy_process_group,all_reduce, ReduceOp, get_rank, get_world_size)
from src.data_loader import load_dataset
import config
import src.mnist as mnist

total_computation_time = 0.0
fault_p = config.FAULT_P

def distributed_setup():
    init_process_group(backend="gloo", init_method="env://")

class Trainer:
    def __init__(self, model, data, test_data,local_epochs, batch_size, save_every, snapshot_path):
        self.device = torch.device("cpu")
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.model = model
        self.data = data
        self.test_data = test_data
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.save_every = save_every
        self.snapshot_path = snapshot_path

    def average_parameters(self):
        for idx in range(len(self.model.weights)):
            w = torch.tensor(self.model.weights[idx], dtype=torch.float64, device=self.device)
            all_reduce(w, op=ReduceOp.SUM)
            w /= self.world_size
            self.model.weights[idx] = w.cpu().numpy()
            b = torch.tensor(self.model.biases[idx], dtype=torch.float64, device=self.device)
            all_reduce(b, op=ReduceOp.SUM)
            b /= self.world_size
            self.model.biases[idx] = b.cpu().numpy()

    def local_train(self):
        self.model.SGD(
            training_data=self.data,
            epochs=self.local_epochs,
            mini_batch_size=self.batch_size,
            eta=config.ETA
        )
        self.average_parameters()
        losses = []
        for x, y in self.data:
            a = self.model.feedforward(x.reshape(-1, 1))
            diff = a - y.reshape(-1, 1)
            losses.append(0.5 * np.mean(diff ** 2))
        return float(np.mean(losses))

    def train(self, total_rounds, start_round=0):
        global total_computation_time
        for r in range(start_round, total_rounds):
            if fault_p > 0.0 and random.random() < fault_p:
                print(f"[worker {self.rank}] Injecting fault at round={r}")
                os._exit(1)

            t0 = time.perf_counter()
            avg_loss = self.local_train()
            t1 = time.perf_counter()
            comp_time = t1 - t0
            total_computation_time += comp_time

            if self.rank == 0:
                correct = self.model.evaluate(self.test_data)
                accuracy = correct / len(self.test_data) if len(self.test_data) > 0 else 0.0

            if (r + 1) % self.save_every == 0:
                path = f"{self.snapshot_path}_round{r+1}.pt"
                torch.save({
                    "weights": [torch.from_numpy(w) for w in self.model.weights],
                    "biases":  [torch.from_numpy(b) for b in self.model.biases],
                    "round":    r + 1
                }, path)
                if self.rank == 0:
                    print(f"[Round {r+1}] Avg Loss: {avg_loss:.6f} | Accuracy: {accuracy * 100:.2f}% | Comp Time: {comp_time:.4f}s | Saved checkpoint {path}")

def main():
    TIMING_CSV = os.path.join(os.getcwd(), "src/pytorch/timings/epoch_timings.csv")
    if not os.path.exists(TIMING_CSV):
        with open(TIMING_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["system_overhead", "avg_comp_time_per_worker"])

    parser = argparse.ArgumentParser(description="Distributed MNIST")
    parser.add_argument("--rounds", type=int, default=config.N_EPOCHS)
    parser.add_argument("--local_epochs", type=int, default=config.SGD_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.MINI_BATCH_SIZE)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--snapshot_path", type=str, default="src/pytorch/snapshot")
    args = parser.parse_args()

    distributed_setup()
    rank = get_rank()
    world_size = get_world_size()

    partitions, test_data = load_dataset()
    data = partitions[rank]
    if config.TRAIN_SAMPLE_SIZE > 0:
        data = data[: config.TRAIN_SAMPLE_SIZE]

    model = mnist.Network(config.NETWORK_ARCHITECTURE)

    ckpts = sorted(glob.glob(f"{args.snapshot_path}_round*.pt"))
    if ckpts:
        info = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        model.weights = [t.cpu().numpy() for t in info["weights"]]
        model.biases  = [t.cpu().numpy() for t in info["biases"]]
        start_round = info.get("round", 0)
        if start_round >= args.rounds and rank == 0:
            print(f"All rounds completed (round {start_round}). Exiting.")
            destroy_process_group()
            return
    else:
        start_round = 0

    run_start = datetime.now()
    if rank == 0:
        print(f"Started training at {run_start.isoformat()}")
        print(f"Workers={world_size}, Samples/worker={len(data)}, "
              f"Batch={args.batch_size}, Local epochs={args.local_epochs}")

    trainer = Trainer(
        model=model,
        data=data,
        test_data=test_data,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        save_every=args.save_every,
        snapshot_path=args.snapshot_path
    )
    trainer.train(args.rounds, start_round)

    comp_tensor = torch.tensor(total_computation_time, dtype=torch.float64)
    all_reduce(comp_tensor, op=ReduceOp.SUM)
    avg_comp_time_per_worker = comp_tensor.item() / world_size
    ckpts = sorted(glob.glob(f"{args.snapshot_path}_round*.pt"))
    if rank == 0:
        run_end = datetime.now()
        total_runtime = (run_end - run_start).total_seconds()
        system_overhead = total_runtime - avg_comp_time_per_worker
        with open(TIMING_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{system_overhead:.6f}", f"{avg_comp_time_per_worker:.6f}"])
            
        for snap in ckpts:
            try:
                os.remove(snap)
                print(f"Deleted snapshot: {snap}")
            except OSError as e:
                print(f"Error deleting {snap}: {e}")

    destroy_process_group()

if __name__ == "__main__":
    main()
