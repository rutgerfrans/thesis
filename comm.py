import mnist
import config

class BaseFederatedCommunicator:
    def broadcast_model(self, model, test_data):
        raise NotImplementedError

    def distribute_data(self, partitions):
        raise NotImplementedError

    def collect_models(self):
        raise NotImplementedError
    
    @staticmethod
    def update_model(model, models, data_sizes):
        model = mnist.Network(config.NETWORK_ARCHITECTURE)
        
        total_size = sum(data_sizes)
        combined_biases = []
        combined_weights = []
        for layer in range(len(models[0].biases)):
            wb = sum(model.biases[layer] * data_sizes[i] for i, model in enumerate(models)) / total_size
            ww = sum(model.weights[layer] * data_sizes[i] for i, model in enumerate(models)) / total_size
            combined_biases.append(wb)
            combined_weights.append(ww)

        model.biases, model.weights = combined_biases, combined_weights
        return model
