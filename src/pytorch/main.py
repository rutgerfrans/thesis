import os
import re
import glob
import argparse
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


def ddp_setup():
    # Initialize process group (requires env vars MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK)
    init_process_group(backend="gloo", init_method="env://")

class MNISTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = torch.tensor(x, dtype=torch.float32).view(1,28,28)
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
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                # convert labels to one-hot
                target = F.one_hot(y, num_classes=logits.size(-1)).float()
                loss = 0.5 * F.mse_loss(logits, target, reduction='mean')
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss / (len(self.dataloader) * self.local_epochs)

    def average_parameters(self):
        for param in self.model.module.parameters():
            all_reduce(param.data, op=ReduceOp.SUM)
            param.data /= self.world_size

    def train(self, total_rounds, start_round=0):
        for r in range(start_round, total_rounds):
            avg_loss = self.local_train()
            self.average_parameters()
            global_round = r + 1
            if self.rank == 0 and global_round % self.save_every == 0:
                ckpt = {
                    'model': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'round': global_round
                }
                path = f"{self.snapshot_path}_round{global_round}.pt"
                torch.save(ckpt, path)
                print(f"[Round {global_round}] Avg Loss: {avg_loss:.6f} | Saved checkpoint {path}")


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
    parser = argparse.ArgumentParser("PyTorch DDP Federated MNIST")
    parser.add_argument('--rounds',        type=int,   default=config.N_EPOCHS,                   help='Global federated rounds')
    parser.add_argument('--local_epochs',  type=int,   default=config.SGD_EPOCHS,                  help='Local SGD epochs per round')
    parser.add_argument('--batch_size',    type=int,   default=config.MINI_BATCH_SIZE,help='Batch size')
    parser.add_argument('--save_every',    type=int,   default=1,                   help='Checkpoint every N rounds')
    parser.add_argument('--snapshot_path', type=str,   default='src/pytorch/snapshot',          help='Checkpoint prefix')
    args = parser.parse_args()

    ddp_setup()
    # load data, model, optimizer
    dataset, model, optimizer = load_train_objs()
    dataloader = prepare_dataloader(dataset, args.batch_size)

    world_size = get_world_size()
    rank = get_rank()
    samples_per_worker = len(dataset)
    import math
    batches_per_worker = math.ceil(samples_per_worker / args.batch_size)
    steps_per_round    = batches_per_worker * args.local_epochs

    if rank == 0:
        print(f"num_workers={world_size}  "
              f"samples_per_worker={samples_per_worker}  "
              f"batches_per_worker={batches_per_worker}  "
              f"steps_per_round={steps_per_round}")

    # find last checkpoint
    ckpts = sorted(glob.glob(f"{args.snapshot_path}_round*.pt"))
    if ckpts:
        last = ckpts[-1]
        info = torch.load(last, map_location='cpu')
        model.load_state_dict(info['model'])
        optimizer.load_state_dict(info['optimizer'])
        start_round = info.get('round', 0)
        if start_round >= args.rounds:
            if get_rank() == 0:
                print(f"All rounds already completed (round {start_round}). Exiting.")
            destroy_process_group()
            return
    else:
        start_round = 0

    trainer = FederatedTrainer(
        model, dataloader, optimizer,
        local_epochs=args.local_epochs,
        save_every=args.save_every,
        snapshot_path=args.snapshot_path
    )
    trainer.train(args.rounds, start_round)
    destroy_process_group()

if __name__ == '__main__':
    main()
