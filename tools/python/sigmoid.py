import numpy as np
import math
import matplotlib.pyplot as plt


def sigmoid(x):
    y = 1.0 / (1 + math.exp(-x))
    return y

arguments = np.linspace(-10, 10, 100)
values = []


for argument in arguments:
    values.append(sigmoid(argument))

plt.title("Sigmoid")
plt.axhline(y = 0, color = "black", linewidth = 1, linestyle = "--")
plt.axhline(y = 1.0, color = "black", linewidth = 1, linestyle = "--")
plt.axhline(y = 0.5, color = "grey", linewidth = 1, linestyle=":")
plt.axvline(color = "grey")
plt.axline((0, 0.5), slope = 0.25, color = "grey", linewidth = 1, linestyle = (0, (5, 5)))
plt.xlabel("Arguments")
plt.ylabel("Values")
plt.grid()
plt.plot(arguments, values, color = "teal", linewidth = 1.5, label = r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.show()
