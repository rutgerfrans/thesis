from mpi4py import MPI
import src.mnist as mnist
import config
from src.data_loader import load_dataset

class MPIFederatedCommunicator():
    def __init__(self):
        self.mpi_comm = MPI.COMM_WORLD
        self.data = None
        self.local_model = None

    def distribute_data(self, data):
        self.data = self.mpi_comm.scatter(data, root=0)
        if self.data[0] is not None:
            self.data[1].SGD(self.data[0], config.SGD_EPOCHS, config.MINI_BATCH_SIZE, config.ETA)
            self.local_model = self.data[1] 
    def collect_models(self):
        return self.mpi_comm.gather(self.local_model, root=0)
    
    def create_data_stack(self, global_model, partitions):
        return [[None, None]] + [(part, global_model) for part in partitions]
    
    def update_model(self, models, partitions, test_set, epoch):
        if self.mpi_comm.Get_rank() == 0:
            if None in models: models.remove(None)
            model = mnist.update_model(models, partitions)
            print(f"Epoch {epoch}/{config.N_EPOCHS - 1} Final Model Evaluation: {model.evaluate(test_set)} / {len(test_set)}")
            return model

    def run(self):
        train_partitions, test_set = load_dataset()
        gm = mnist.Network(config.NETWORK_ARCHITECTURE)

        for epoch in range(config.N_EPOCHS):
            self.distribute_data(self.create_data_stack(gm, train_partitions))
            gm = self.update_model(self.collect_models(), train_partitions, test_set, epoch)