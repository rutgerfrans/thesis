import src.mnist as mnist
import config
from src.comms.comm import BaseFederatedCommunicator

class MPIFederatedCommunicator(BaseFederatedCommunicator):
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
    
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
    
    def worker(self, data, epoch, test_data):
        local_model = self.train_local_model(data[1], data[0], test_data)
        return local_model
