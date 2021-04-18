from network import Network
from loader import load_data_wrapper


training_data, validation_data, test_data = load_data_wrapper()

# print(training_data)

# network = Network([2, 10, 10])
network = Network([784, 100, 10])

network.stochastic_gradient_descend(training_data, 50, 10, 1.2, test_data)
# network.gradient_descend([([1, 2], [1,2,3,4,5,6,7,8,9,0])], 3.0)
