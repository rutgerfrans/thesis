
import random
import config

class test():
    def __init__(self, crash_at_epoch, num_of_workers_crash):
        self.crash_at_epoch = crash_at_epoch
        self.nodes_to_be_crashed = set(random.sample(range(config.N_PARTITIONS), num_of_workers_crash))
        print("Crashing ", num_of_workers_crash, " workers at epochs ", crash_at_epoch)
        print("===============================")

    def simulate(self, epoch, processes):
        if epoch in self.crash_at_epoch:
            for i, proc in enumerate(processes):
                if i in self.nodes_to_be_crashed:
                    self.crash_worker(proc)

    def crash_worker(self, proc):
        proc.kill()