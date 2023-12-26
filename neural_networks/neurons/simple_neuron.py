class Neuron:
    def __init__(self, weights, bias = 0):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]
        weighted_sum += self.bias
        return self.activation_function(weighted_sum)

    def activation_function(self, x):
        return 1 if x > 1 else 0
