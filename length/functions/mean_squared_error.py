import numpy as np

from length.function import Function


class MeanSquaredError(Function):
    """
    This function calculates the Mean Squared Error between two given vectors, as described in
    https://en.wikipedia.org/wiki/Mean_squared_error
    """
    name = "MeanSquaredError"

    def __init__(self):
        super().__init__()
        # TODO: add more initialization if necessary

    @staticmethod
    def create_one_hot(data, shape):
        assert len(shape) == 2, "Providing integers as second input to MSE only works with two dimensional input vectors"
        # TODO: create a one-hot representation out of the given label vector
        # Example: assume input is the following: [2, 3], and shape is (2, 4)
        # the resulting vector should look like this:
        # result = [[0, 0, 1, 0], [0, 0, 0, 1]]
        # see: https://stackoverflow.com/questions/29831489/numpy-1-hot-array
        one_hot = np.zeros(shape)
        one_hot[np.arange(shape[0]), data] = 1

        assert one_hot.shape == shape, "one_hot shape does not fit"
        return one_hot

    def internal_forward(self, inputs):
        x1, x2 = inputs

        if np.issubdtype(x2.dtype, np.integer):
            x2 = self.create_one_hot(x2, x1.shape)

        # TODO: calculate the mean squared error of x1 and x2
        error = np.array([np.mean(np.sum(np.square(x1i - x2i))) for x1i, x2i in zip(x1, x2)])
        print("rmse: " + str(error))

        return error,

    def internal_backward(self, inputs, gradients):
        x1, x2 = inputs
        gx, = gradients

        # TODO: calculate the gradients of this function with respect to its inputs
        if np.issubdtype(x2.dtype, np.integer):
            x2 = self.create_one_hot(x2, x1.shape)

        print("backward input x1.shape: " + str(x1.shape))
        print("backward input x2.shape: " + str(x2.shape))

        gradient_1 = np.array([2 * np.mean(np.sum(x1i - x2i)) for x1i, x2i in zip(x1, x2)]) # d MSE / d x1
        gradient_2 = np.array([2 * np.mean(np.sum(x2i - x1i)) for x1i, x2i in zip(x1, x2)]) # d MSE / d x2

        print("grad x1: " + str(gradient_1))
        print("grad x2: " + str(gradient_2))
        if np.issubdtype(x2.dtype, np.integer):
            # in case we used MSE as loss function, we won't propagate any gradients to the labels
            return gradient_1, None

        return gradient_1, gradient_2


def mean_squared_error(input_1, input_2):
    """
    This function calculates the Mean Squared Error between input_1 and input_2. Both inputs should be vectors of the
    same shape. You can also supply a one-dimensional list of integers.
    If you do so this vector will be converted to a one_hot representation that fits to the shape of the second
    input
    :param input_1: the first vector of any shape
    :param input_2: the second vector. Needs to have the same shape as the first vector, or be a one-dimensional int vector
    :return: the mean squared error between input_1 and input_2
    """
    return MeanSquaredError()(input_1, input_2)
