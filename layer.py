import abc
import math
import numpy as np
from random import sample
from utilities import xavierInit, getOutputSize, relu


class Layer(abc.ABC):
    pass


class Conv(Layer):

    def __init__(self, filterSize: int, numFilters: int, inputSize: tuple, padding: int, stride: int, first=False):
        """

        :param filterSize: width/height dimension of filters
        :param numFilters: Number of filters, each with dimension filterSize x filterSize x inputChannels
        :param inputSize: Size of the input matrix in format side size x side size x channels
        :param padding: Zero padding to be added on axis 0 and 1 of input matrix
        :param stride: Distance moved by filter as it goes over the input matrix
        :param first: True if it is the first layer in the network
        """
        self.filterSize = filterSize
        self.numFilters = numFilters
        self.inputChannels = inputSize[2]
        self.outputSize = getOutputSize(inputSize,
                                        tuple([filterSize, filterSize, self.numFilters]),
                                        padding,
                                        stride)
        # Parameter based initialization
        self.filters = xavierInit(filterSize*filterSize*numFilters,
                                  tuple([self.numFilters, filterSize, filterSize, self.inputChannels]),
                                  first)
        self.bias = np.zeros(numFilters)
        self.padding = padding
        self.stride = stride
        self.lastOutput = np.zeros(self.outputSize)
        # Used for backprop
        self.lastInput = None


    def feed(self, inputMatrix: np.ndarray):
        """
        Performs convolution operation the input values and outputs the resulting matrix for the next layer
        :param inputMatrix: Matrix to be convoluted
        :return: ndarray of convoluted values
        """

        self.lastInput = inputMatrix
        inMatrix = inputMatrix

        if self.padding != 0:
            inMatrix = np.asarray([np.pad(x,
                                          self.padding,
                                          mode='constant',
                                          constant_values=0) for x in inputMatrix[:, :]])

        sideSize = self.outputSize[1]
        # for each filter
        for iFilter in range(self.numFilters):
            curFilter = self.filters[iFilter]
            # For each index in output
            for row in range(sideSize):
                startRow = row * self.stride
                endRow = startRow + self.filterSize
                for column in range(sideSize):
                    # Sets indices for current receptive field
                    startColumn = column * self.stride
                    endColumn = startColumn + self.filterSize
                    curSection = inMatrix[startRow:endRow, startColumn:endColumn, :]
                    # Filters are same size as the receptive field
                    self.lastOutput[row, column, iFilter] = np.multiply(curSection, curFilter).sum() + self.bias[iFilter]


class Relu(Layer):
    def __init__(self, alpha: float):
        """

        :param alpha: Constant value for leaky Relu
        """
        self.alpha = alpha
        # Vectorized Relu function
        self.vRelu = relu(alpha)
        self.lastOutput = None
        self.weights = None

    def feed(self, inMatrix: np.ndarray):
        self.lastOutput = self.vRelu(inMatrix)

        # Each location is one if the value was larger than 0, alpha otherwise
        self.weights = (self.lastOutput > 0).astype(float)
        self.weights[self.weights == 0] = self.alpha
     
     
class FC(Layer):
    def __init__(self, numNodes: int, dropout: float, numInputs: int):
        """

        :param numNodes: Number of nodes in the layer, where each has # weights = numInputs
        :param dropout: Dropout percentage, any number [0,1]
        :param numInputs: Length of array inputted from the layer before
        """
        self.nodes = np.random.normal(0, math.sqrt(2/(numNodes * numInputs)), (numNodes, numInputs))
        self.biases = np.zeros(numNodes)
        self.numNodes = numNodes
        self.numUsedNodes = int(self.numNodes * (1-dropout))
        self.numDroppedNodes = self.numNodes - self.numUsedNodes
        self.nodeIndex = [x for x in range(self.numNodes)]
        self.numInputs = numInputs
        self.lastOutput = np.zeros(self.numNodes)
        self.lastInput = None

        self.vSigmoid = np.vectorize(self._sigmoid)

    def _sigmoid(self, x):
        # To avoid overflow, ensure math.exp is never called with a positive value
        # https://stackoverflow.com/questions/36268077/overflow-math-range-error-for-log-or-exp?rq=1
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        else:
            return 1 / (1 + math.exp(-x))

    def feed(self, inputMatrix: np.ndarray):
        self.lastInput = inputMatrix
        # Randomly selected indices to drop
        nodesToDrop = sample(self.nodeIndex, self.numDroppedNodes)
        nodesToDrop.sort()
        for iNode, node in enumerate(self.nodes):
            out = np.multiply(inputMatrix, node).sum()
            out += self.biases[iNode]
            self.lastOutput[iNode] = out
        if self.numDroppedNodes > 0:
            self.lastOutput[nodesToDrop] = 0
        self.lastOutput = self.vSigmoid(self.lastOutput)


class Pool(Layer):
    def __init__(self, filterSize: int, stride: int, inDim: tuple):
        """

        :param filterSize: Width/height of max pooling receptive field
        :param stride: How far the receptive field moves each iteration
        :param inDim:  Dimensions of input matrix (side size x side size x channels)
        """
        self.filterSize = filterSize
        self.stride = stride
        self.inDim = inDim
        self.channels = inDim[-1]
        self.outputSize = getOutputSize(self.inDim, 
                                        tuple([self.filterSize, self.filterSize, self.channels]),
                                        0,
                                        self.stride)

        self.lastOutput = np.zeros(self.outputSize)
        # Used for backpropogation
        self.weights = np.zeros(self.inDim)

    def feed(self, inLayer: np.ndarray):

        # We have to use the naive implementation in order to store the location
        # of selected max values
        # Could probably be improved but was not sure how to do efficiently
        # so we chose to have full accuracy with a slow program
        for channel in range(self.channels):
            for row in range(self.outputSize[0]):
                startRow = self.stride * row
                endRow = startRow + self.filterSize
                for col in range(self.outputSize[1]):
                    startCol = self.stride * col
                    endCol = startCol + self.filterSize
                    curMax = inLayer[startRow, startCol, channel]
                    # Current max indices
                    mR, mC = startRow,startCol
                    for r in range(startRow, endRow):
                        for c in range(startCol, endCol):
                            self.weights[r, c, channel] = 0
                            cur = inLayer[r, c, channel]
                            if cur > curMax:
                                curMax = cur
                                mR, mC = r, c
                    self.lastOutput[row, col, channel] = curMax
                    self.weights[mR, mC] = 1
