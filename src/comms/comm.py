import src.mnist as mnist
import config
import src.utils as utils
from src.data_loader import load_dataset

class BaseFederatedCommunicator:
    def distribute_data(self):
        raise NotImplementedError
    
    def collect_models(self):
        raise NotImplementedError
    
    def create_partition_files(self, partitions):
        self.partition_files = []
        for i, part in enumerate(partitions):
            utils.save_pickle(part, self.partition_file+f"{i}.pkl")
            self.partition_files.append(self.partition_file+f"{i}.pkl")
        return self.partition_files
    
    def create_data_stack(self, global_model, partitions):
        return [[part, global_model] for part in self.create_partition_files(partitions)]
    
    def init_global_model(self):
        net = mnist.Network(config.NETWORK_ARCHITECTURE)
        return net
    
    def train_model(self, global_model, local_data):
        net = mnist.Network(global_model.sizes)
        net.biases = [b.copy() for b in global_model.biases]
        net.weights = [w.copy() for w in global_model.weights]
        net.SGD(local_data, epochs=config.SGD_EPOCHS, mini_batch_size=config.MINI_BATCH_SIZE, eta=config.ETA)
        return net
    
    def update_model(self, models, partitions, test_set, epoch):
        data_sizes = [len(p) for p in partitions]
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
    
    def run(self):
        train_partitions, test_set = load_dataset()
        gm = self.init_global_model()

        for epoch in range(config.N_EPOCHS):
            self.distribute_data(self.create_data_stack(gm, train_partitions), epoch)
            gm = self.update_model(self.collect_models(), train_partitions, test_set, epoch)