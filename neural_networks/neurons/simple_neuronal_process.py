input_data = [0.8, 0.5]

weights = [0.2, 0.6]

bias = 1.5

# The weighted sum of inputs and weights with adding bias
weighted_sum = 0
for i in range(len(input_data)):
    weighted_sum += input_data[i] * weights[i]
weighted_sum += bias

# Activation function (in this case, using a simple threshold function)
def activation_function(x):
    return 1 if x > 1 else 0

# Applying the activation function to the weighted sum
output = activation_function(weighted_sum)

# Printing the output
print("Output:", output)
