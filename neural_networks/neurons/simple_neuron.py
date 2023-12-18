class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, input_data):
        weighted_sum = 0
        for i in range(len(input_data)):
            weighted_sum += input_data[i] * self.weights[i]
        weighted_sum += self.bias
        return self.activation_function(weighted_sum)

    def activation_function(self, x):
        return 1 if x > 1 else 0
