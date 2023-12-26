import numpy as np
import matplotlib.pyplot as plt
from simple_neuron import Neuron


def compute(input_values, weights):
    neuron = Neuron(weights)
    outputs = {}
    for x_1 in input_values:
        outputs[x_1] = {}
        for x_2 in input_values:
            outputs[x_1][x_2] = neuron.forward(inputs = [x_1, x_2])

    return outputs

def present(input_values, outputs):
    for x_1 in input_values:
        for x_2 in input_values:
            print(f"{x_1}\t{x_2}\t", outputs[x_1][x_2])

def analyse(input_values, weights):
    outputs = compute(input_values, weights)
    present(input_values, outputs)

def describe(input_values, weights):
    print(f"Weights: {weights[0]}, {weights[1]}:\n")
    analyse(input_values, weights)
    print()

def plot(weights, inputs):
    outputs = []
    for weight in weights:
        neuron = Neuron(weights = [weight, weight])
        outputs.append(neuron.forward(inputs))
    plt.title(f"x: weights, y: output (input: {inputs})")
    plt.plot(weights, outputs)
    plt.show()


input_values = [-1, 0, 1]

describe(input_values, weights = [1, 1])
describe(input_values, weights = [0.9, 0.9])
describe(input_values, weights = [0.5, 0.5])
describe(input_values, weights = [0.4, 0.4])
describe(input_values, weights = [-1, -1])
describe(input_values, weights = [-0.9, -0.9])
describe(input_values, weights = [-0.5, -0.5])
describe(input_values, weights = [-0.4, -0.4])

weights = np.arange(-0.9, 0.9, 0.01)

plot(weights, inputs = [1, 1])
plot(weights, inputs = [1, -1])
plot(weights, inputs = [-1, 1])
plot(weights, inputs = [-1, -1])
