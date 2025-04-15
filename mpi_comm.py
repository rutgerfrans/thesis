from mpi4py import MPI #type: ignore
import mnist
import config
from comm import BaseFederatedCommunicator

class MPIFederatedCommunicator(BaseFederatedCommunicator):
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()

    def broadcast_model(self, model, test_data):
        model = self.comm.bcast(model, root=0)
        test_data = self.comm.bcast(test_data, root=0)
        return model, test_data

    def distribute_data(self, partitions):
        for i in range(config.N_PARTITIONS):
            self.comm.send(partitions[i], dest=i + 1, tag=11)

    def collect_models(self, n_partitions):
        models = []
        for i in range(1, n_partitions + 1):
            models.append(self.comm.recv(source=i, tag=22))
        return models

    def train_local_model(self, global_model, local_data, test_data):
        if global_model is not None:
            net = mnist.Network(global_model.sizes)
            net.biases = [b.copy() for b in global_model.biases]
            net.weights = [w.copy() for w in global_model.weights]
        else:
            net = mnist.Network(config.NETWORK_ARCHITECTURE)
        net.SGD(local_data, epochs=config.SGD_EPOCHS, mini_batch_size=config.MINI_BATCH_SIZE,
                eta=config.ETA, test_data=test_data)
        return net

    def worker_epoch(self, epoch):
        model, test_data = self.broadcast_model(None, None)
        if config.DEBUG:
            print(f"[DEBUG] Worker {self.rank}: Epoch {epoch}: Received model and test data.", flush=True)
        local_data = self.comm.recv(source=0, tag=11)
        if config.DEBUG:
            print(f"[DEBUG] Worker {self.rank}: Epoch {epoch}: Received local data.", flush=True)
        updated_model = self.train_local_model(model, local_data, test_data)
        self.comm.send(updated_model, dest=0, tag=22)
        if config.DEBUG:
            print(f"[DEBUG] Worker {self.rank}: Epoch {epoch}: Sent updated model.", flush=True)
