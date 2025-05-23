import os, glob, argparse, time, csv, random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp, get_rank, get_world_size
from src.data_loader import load_dataset
import config

total_computation_time = 0.0

fault_p = config.FAULT_P

def ddp_setup():
    init_process_group(backend="gloo", init_method="env://")

class MNISTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = torch.tensor(x, dtype=torch.float32).view(1, 28, 28)
        if isinstance(y, (np.ndarray, list)) and len(y) == 10:
            y = int(np.argmax(y))
        y = torch.tensor(y, dtype=torch.long)
        return x, y

class MNISTTorchNet(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.normal_(m.bias)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class FederatedTrainer:
    def __init__(self, model, dataloader, optimizer, local_epochs, save_every, snapshot_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = get_rank()
        self.world_size = get_world_size()
        model = model.to(self.device)
        self.model = DDP(model)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.local_epochs = local_epochs
        self.save_every = save_every
        self.snapshot_path = snapshot_path

    def local_train(self):
        self.model.train()
        total_loss = 0.0
        for epoch in range(self.local_epochs):
            self.dataloader.sampler.set_epoch(epoch)
            for x, y in self.dataloader:
                self.optimizer.zero_grad()
                logits = self.model(x.to(self.device))
                target = F.one_hot(y.to(self.device), logits.size(-1)).float()
                loss = 0.5 * F.mse_loss(logits, target, reduction="mean")
                loss.backward()
                self.optimizer.step()
                self.average_parameters()
                total_loss += loss.item()
        avg_loss = total_loss / (len(self.dataloader) * self.local_epochs)
        return avg_loss

    def average_parameters(self):
        for param in self.model.module.parameters():
            all_reduce(param.data, op=ReduceOp.SUM)
            param.data /= self.world_size

    def train(self, total_rounds, start_round=0):
        global total_computation_time
        for r in range(start_round, total_rounds):
            if fault_p > 0.0 and random.uniform(0, 1) < fault_p:
                print(f"[worker {self.rank}] Injecting fault at round={r}")
                os._exit(1)

            # Measure computation time for this worker for one global round
            t0 = time.perf_counter()
            avg_loss = self.local_train()
            self.average_parameters()
            t1 = time.perf_counter()
            comp_time = t1 - t0

            # Accumulate into global counter
            total_computation_time += comp_time

            if self.rank == 0 and (r + 1) % self.save_every == 0:
                path = f"{self.snapshot_path}_round{r+1}.pt"
                torch.save({"model": self.model.module.state_dict(),"optimizer": self.optimizer.state_dict(),"round": r + 1}, path)
                print(f"[Round {r+1}] Avg Loss: {avg_loss:.6f} | Comp Time: {comp_time:.4f}s | Saved checkpoint {path}")


def load_train_objs():
    partitions, _ = load_dataset()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    dataset = MNISTDataset(partitions[rank])
    model = MNISTTorchNet(config.NETWORK_ARCHITECTURE)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.ETA)
    return dataset, model, optimizer


def prepare_dataloader(dataset, batch_size):
    sampler = DistributedSampler(dataset, shuffle=False)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def main():
    TIMING_CSV = os.path.join(os.getcwd(), "src/pytorch/timings/epoch_timings.csv")
    if not os.path.exists(TIMING_CSV):
        with open(TIMING_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "system_overhead",
                "avg_comp_time_per_worker"
            ])

    parser = argparse.ArgumentParser("PyTorch DDP Federated MNIST")
    parser.add_argument("--rounds", type=int, default=config.N_EPOCHS)
    parser.add_argument("--local_epochs", type=int, default=config.SGD_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.MINI_BATCH_SIZE)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--snapshot_path", type=str, default="src/pytorch/snapshot")
    args = parser.parse_args()

    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    dataloader = prepare_dataloader(dataset, args.batch_size)

    world_size = get_world_size()
    rank = get_rank()

    run_start = datetime.now()
    if rank == 0:
        print(f"Started training at {run_start.isoformat()}")

    if rank == 0:
        print(f"Workers={world_size}, Samples/worker={len(dataset)}, Batch={args.batch_size}, Local epochs={args.local_epochs}")

    ckpts = sorted(glob.glob(f"{args.snapshot_path}_round*.pt"))
    if ckpts:
        info = torch.load(ckpts[-1], map_location="cpu")
        model.load_state_dict(info["model"])
        optimizer.load_state_dict(info["optimizer"])
        start_round = info.get("round", 0)
        if start_round >= args.rounds and rank == 0:
            print(f"All rounds completed (round {start_round}). Exiting.")
            destroy_process_group()
            return
    else:
        start_round = 0

    trainer = FederatedTrainer(model, dataloader, optimizer,args.local_epochs, args.save_every, args.snapshot_path)
    trainer.train(args.rounds, start_round)

    # After training, gather and average computation times across workers
    comp_tensor = torch.tensor(total_computation_time, dtype=torch.float64)
    all_reduce(comp_tensor, op=ReduceOp.SUM)
    avg_comp_time_per_worker = comp_tensor.item() / world_size

    if rank == 0:
        run_end = datetime.now()
        total_runtime = (run_end - run_start).total_seconds()
        system_overhead = total_runtime - avg_comp_time_per_worker
        #print(f"Completed training at {run_end.isoformat()}")
        #print(f"Avg comp time per worker: {avg_comp_time_per_worker:.4f}s; System overhead: {system_overhead:.4f}s")

        with open(TIMING_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{system_overhead:.6f}",
                f"{avg_comp_time_per_worker:.6f}"
            ])
    if rank == 0:
        snapshots = glob.glob(f"{args.snapshot_path}_round*.pt")
        for snap in snapshots:
            try:
                os.remove(snap)
                print(f"Deleted snapshot: {snap}")
            except OSError as e:
                print(f"Error deleting {{snap}}: {{e}}")
    destroy_process_group()

if __name__ == "__main__":
    main()
