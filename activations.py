import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def backprop_sigmoid_unit(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def backprop_relu_unit(z):
    dz = np.ones(z.shape)
    dz[z <= 0] = 0
    return dz

def softmax(z):
    norm_z = z - np.max(z, axis=0, keepdims=True)
    exp_values = np.exp(norm_z)
    value = exp_values / np.sum(exp_values, axis=0, keepdims=True)
    # print(value.shape)
    # print(np.sum(value, axis=0))
    return value
