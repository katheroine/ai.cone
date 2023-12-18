import numpy as np
import matplotlib.pyplot as plt
from simple_neuron import Neuron


inputs = np.arange(-10, 10, 0.1)
outputs = []

for input in inputs:
    # Create an instance of the Neuron class
    neuron = Neuron(weights=[0.2, 0.6], bias=1.5)

    # Define the input data
    input_data = [0.8 * input, 0.5 * input]

    # Get the output from the neuron
    output = neuron.forward(input_data)

    # Print the output
    # print("Output:", output)
    outputs.append(output)


plt.plot(inputs, outputs)
plt.title('Input/output characteristic')
plt.show()

