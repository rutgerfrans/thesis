from src.comms.comm import BaseFederatedCommunicator
from mpi4py import MPI #type: ignore

class MPIFederatedCommunicator(BaseFederatedCommunicator):
    def __init__(self):
        self.mpi_comm = MPI.COMM_WORLD
        self.data = None
        self.local_model = None

    def distribute_data(self, data, epoch):
        self.data = self.mpi_comm.scatter(data, root=0)
        if self.data[0] is not None:
            self.local_model = self.train_model(self.data[1], self.data[0])

    def collect_models(self): # maybe create a try except here
        return self.mpi_comm.gather(self.local_model, root=0)
    
    def create_data_stack(self, global_model, partitions):
        return [[None, None]] + [(part, global_model) for part in partitions]
    
    def update_model(self, models, partitions, test_set, epoch):
        if self.mpi_comm.Get_rank() == 0:
            if None in models: models.remove(None)
            return super().update_model(models, partitions, test_set, epoch)
    
