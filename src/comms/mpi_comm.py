from src.comms.comm import BaseFederatedCommunicator
from mpi4py import MPI #type: ignore
import src.mnist as mnist
import config

class MPIFederatedCommunicator(BaseFederatedCommunicator):
    def __init__(self):
        self.mpi_comm = MPI.COMM_WORLD
        self.data = None
        self.local_model = None

    def distribute_data(self, data):
        self.data = self.mpi_comm.scatter(data, root=0)
        if self.data[0] is not None:
            self.local_model = self.train_model(self.data[1], self.data[0])

    def collect_models(self):
        return self.mpi_comm.gather(self.local_model, root=0)
    
    def create_data_stack(self, global_model, partitions):
        return [[None, None]] + [(part, global_model) for part in partitions]
    
    def update_model(self, models, data_sizes, test_set, epoch):
        if self.mpi_comm.Get_rank() == 0:
            if None in models: models.remove(None)
            total_size = sum(data_sizes)
            combined_biases = []
            combined_weights = []
            for layer in range(len(models[0].biases)):
                wb = sum(model.biases[layer] * data_sizes[i] for i, model in enumerate(models)) / total_size
                ww = sum(model.weights[layer] * data_sizes[i] for i, model in enumerate(models)) / total_size
                combined_biases.append(wb)
                combined_weights.append(ww)
            model = mnist.Network(models[0].sizes)
            model.biases, model.weights = combined_biases, combined_weights
            print(f"Epoch {epoch}/{config.N_EPOCHS - 1} Final Model Evaluation: {model.evaluate(test_set)} / {len(test_set)}")
            return model