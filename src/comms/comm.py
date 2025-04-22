import src.mnist as mnist
import config

class BaseFederatedCommunicator:
    def distribute_data(self):
        raise NotImplementedError
    
    def collect_models(self):
        raise NotImplementedError
    
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