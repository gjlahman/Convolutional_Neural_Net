import numpy as np
from math import sqrt


def relu(alpha: float):
    """
    Creates a vectorized leaky ReLU activation function.
    Leaky alleviates the problem of vanishing gradient

    :param alpha: user selected hyperparameter
    :return: vectorized function
    """
    def _relu(x):
        return max(alpha*x, x)
    return np.vectorize(_relu)

def xavierInit(n: int, size: tuple, first: bool):
    """
    Intializes weights according to xavier initialization
    https://arxiv.org/pdf/1502.01852.pdf

    :param n: number of units in current layer (1 if it is first)
    :param size: Size of weight array
    :param first: True if first layer in network, otherwise false
    :return: Weight array, which is N~(0,sqrt(2/n))
    """
    if first:
        return np.random.normal(0, 1, size)
    else:
        return np.random.normal(0, sqrt(2/n), size)


def getOutputSize(inDim: tuple, filterSize: tuple, padding: int, stride: int):
    """
    Calculates the size of the convolutional output matrix. Raises value error if it is an invalid size
    :param inDim (side, side, channel)
    :param filterSize ( side, side, channel)
    :return: tuple with size of output
    """
    sideSize = (inDim[1] - filterSize[0] + 2*padding) / stride + 1
    if sideSize.is_integer():
        return tuple([int(sideSize), int(sideSize), filterSize[2]])
    else:
        raise ValueError("Pooling output size must be integer. Adjust filter size, padding, or stride")
