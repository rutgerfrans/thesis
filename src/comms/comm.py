import src.mnist as mnist
import config

class BaseFederatedCommunicator:
    def distribute_data(self):
        raise NotImplementedError
    
    def train_model(self, global_model, local_data):
        if global_model is not None:
            net = mnist.Network(global_model.sizes)
            net.biases = [b.copy() for b in global_model.biases]
            net.weights = [w.copy() for w in global_model.weights]
        else:
            net = mnist.Network(config.NETWORK_ARCHITECTURE)
        net.SGD(local_data, epochs=config.SGD_EPOCHS, mini_batch_size=config.MINI_BATCH_SIZE,
                eta=config.ETA)
        return net

    def collect_models(self):
        raise NotImplementedError
    
    def update_model(self, model, models, data_sizes):
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
    
    def evaluate_model(self, model, test_data, epoch):
        return f"Epoch {epoch}/{config.N_EPOCHS - 1} Final Model Evaluation: {model.evaluate(test_data)} / {len(test_data)}"