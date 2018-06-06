import numpy as np

from length.function import Function


class Dropout(Function):
    name = "Dropout"

    def __init__(self, dropout_ratio):
        super().__init__()
        if not 0.0 <= dropout_ratio < 1:
            raise ValueError("dropout_ratio must be in range [0, 1)")
        self.dropout_ratio = dropout_ratio
        # TODO: add more initialization if necessary
        self.mask = None

    def internal_forward(self, inputs):
        x, = inputs
        # TODO: implement forward pass of dropout function

        # generate mask
        self.mask = np.random.choice([0, 1], x.shape, p=[1 - self.dropout_ratio, self.dropout_ratio])

        # scale output if ratio > 0
        if self.dropout_ratio > 0:
            mask = self.mask / (1 - self.dropout_ratio)
        else:
            mask = self.mask

        return x * mask,

    def internal_backward(self, inputs, gradients):
        gradient, = gradients
        # TODO: implement backward pass of dropout function
        out = self.mask * gradient
        return out,


def dropout(x, dropout_ratio=0.5, train=True):
    """
    This function implements dropout (http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf),
    a regularization method for neural networks.
    :param x: the input vector where parts shall be dropped
    :param dropout_ratio: the ratio of which to perform dropout
    :param train: whether we are currently running in train or testing mode (default: True)
    :return: a vector with a portion of elements zeroed out, this portion is defined by `dropout_ratio`.
    """
    # TODO: call the dropout function and handle the scenario if we are not training
    return Dropout(dropout_ratio)(x) if train else x
