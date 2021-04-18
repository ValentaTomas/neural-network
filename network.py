import numpy as np
from sigmoid import sigmoid, sigmoid_prime


class Network():
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def stochastic_gradient_descend(self, training_data, epochs, batch_size, rate, test_data=None):
        data_size = len(training_data)
        if test_data:
            n_test = len(test_data)
            
        for epoch in range(epochs):
            np.random.shuffle(training_data)

            batches = [training_data[start:(start + batch_size)]
                       for start in range(0, data_size, batch_size)]

            for batch in batches:
                self.update_batch(batch, rate)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch,
                                                    self.evaluate(test_data), n_test))

    def update_batch(self, batch, rate):
        batch_size = len(batch)

        new_weights = [np.zeros(layer_weights.shape)
                       for layer_weights in self.weights]
        new_biases = [np.zeros(layer_biases.shape)
                      for layer_biases in self.biases]

        for x, y in batch:
            delta_weights, delta_biases = self.backpropagate(x, y)
            new_weights = [new_weight + delta_weight for new_weight,
                           delta_weight in zip(new_weights, delta_weights)]
            new_biases = [new_bias + delta_bias for new_bias,
                          delta_bias in zip(new_biases, delta_biases)]

        self.weights = [weights - (rate / batch_size) * delta for weights,
                        delta in zip(self.weights, new_weights)]
        self.biases = [biases - (rate / batch_size) * delta for biases,
                       delta in zip(self.biases, new_biases)]

    def backpropagate(self, inputs, outputs):
        delta_weights = [np.zeros(layer_weights.shape)
                         for layer_weights in self.weights]
        delta_biases = [np.zeros(layer_biases.shape)
                        for layer_biases in self.biases]

        last_activation = inputs
        activations = [inputs]
        zs = []

        for layer_weights, layer_biases in zip(self.weights, self.biases):
            last_output = np.dot(layer_weights, last_activation) + layer_biases
            last_activation = sigmoid(last_output)

            zs.append(last_output)
            activations.append(last_activation)

        last_delta = (activations[-1] - outputs) * sigmoid_prime(zs[-1])

        delta_biases[-1] = last_delta

        delta_weights[-1] = np.dot(last_delta, activations[-2].transpose())

        for layer in range(2, len(self.layer_sizes)):
            z = zs[-layer]
            sp = sigmoid_prime(z)

            last_delta = np.dot(
                self.weights[-layer + 1].transpose(), last_delta) * sp

            delta_biases[-layer] = last_delta
            delta_weights[-layer] = np.dot(last_delta,
                                           activations[- layer - 1].transpose())

        return (delta_weights, delta_biases)
