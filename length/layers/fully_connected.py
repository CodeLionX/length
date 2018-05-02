import numpy as np

from length.layer import Layer
from length.constants import DTYPE
from length.initializers.xavier import Xavier


class FullyConnected(Layer):
    """
    The FullyConnected Layer is one of the base building blocks of a neural network. It computes a weighted sum
    over the input, using a weight matrix. It furthermore applies a bias term to this weighted sum to allow linear
    shifts of the computed values.
    """
    name = "FullyConnected"

    def __init__(self, num_inputs, num_outputs, weight_init=Xavier()):
        super().__init__()
        # HINT: some ideas for the implementation can be found here:
        # https://cs231n.github.io/neural-networks-1/#layers

        # TODO: initialize weights with correct shape, using the weight initializer 'weight_init'
        self._weights = np.empty(shape=(num_inputs, num_outputs), dtype=DTYPE)
        weight_init(self._weights)
        # TODO: initialize bias with correct shape and correct initial values
        self.bias = np.empty(shape=(num_outputs,), dtype=DTYPE)
        weight_init(self.bias)

    @property
    def weights(self):
        # TODO: (if necessary) add code to transform between internal and external representation
        return self._weights

    @weights.setter
    def weights(self, value):
        # TODO: (if necessary) add code to transform between internal and external representation
        self._weights = value

    def internal_forward(self, inputs):
        # x.shape = (n_batches, pix)
        x, = inputs
        # TODO: calculate result for this layer
        # HINT: the expected shape of the result is (usually) not equal to the shape of x

        #print("x.shape: " + str(x.shape))
        print("W.shape: " + str(self._weights.shape))
        #print("b.shape: " + str(self.bias.shape))
        results = [np.dot(xi, self._weights) + self.bias for xi in x]

        result = np.array(results, dtype=DTYPE)
        return result,

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients

        print("backward x.shape: " + str(x.shape))
        # TODO: calculate gradients with respect to inputs for this layer
        grad_x = None # np.array([self._weights for _ in x])
        grad_w = None
        grad_b = None # np.full(self.bias.shape, 1, dtype=DTYPE)

        assert grad_x.shape == x.shape
        # the gradients of the weights should have the same shape as the internal weights array for the test case
        assert grad_w.shape == self._weights.shape
        assert grad_b.shape == self.bias.shape

        return grad_x, grad_w, grad_b

    def internal_update(self, parameter_deltas):
        delta_w, delta_b = parameter_deltas
        # TODO: apply updates to weights and bias according to deltas from optimizer
        self._weights = np.add(self._weights,  delta_w)
        self.bias = np.add(self.bias, delta_b)
