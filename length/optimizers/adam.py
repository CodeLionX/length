import numpy as np

from length.optimizer import Optimizer


class Adam(Optimizer):
    """
    The Adam optimizer (see https://arxiv.org/abs/1412.6980)
    :param learning_rate: initial step size
    :param beta1: Exponential decay rate of the first order moment
    :param beta2: Exponential decay rate of the second order moment
    :param eps: Small value for numerical stability
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.ibeta1 = 1 - beta1
        self.beta2 = beta2
        self.ibeta2 = 1 - beta2
        self.eps = eps

        # TODO: add more initialization code
        self.t = 0
        self.m = None
        self.v = None
        self.isInit = True

    def run_update_rule(self, gradients, layer):
        # TODO: implement Adam update rule as specified in https://arxiv.org/abs/1412.6980
        gradient, = gradients

        if self.isInit:
            self.m = np.full(gradient.shape, 0)
            self.v = np.full(gradient.shape, 0)
            self.isInit = False

        self.t += 1
        self.m = self.beta1 * self.m + self.ibeta1 * gradient
        self.v = self.beta2 * self.v + self.ibeta2 * np.square(gradient)
        m_corr = self.m / (1 - np.power(self.beta1, self.t))
        v_corr = self.v / (1 - np.power(self.beta2, self.t))
        delta = self.learning_rate * m_corr / (np.sqrt(v_corr) + self.eps)
        return delta,
