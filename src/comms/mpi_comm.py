import src.mnist as mnist
import config
from src.comms.comm import BaseFederatedCommunicator

class MPIFederatedCommunicator(BaseFederatedCommunicator):
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()

    def distribute_data(self, data):
        return self.comm.scatter(data, root=0)

    def collect_models(self, local_model):
        return self.comm.gather(local_model, root=0)