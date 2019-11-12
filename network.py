import csv
import numpy as np
from math import log
from time import gmtime, strftime, time
import layer


class Network:

    requiredParams = {'conv': 4, 'fc': 2, 'pool': 2, 'relu': 1}

    def __init__(self, structureFile: str):
        """

        :param structureFile: name of csv file defining network structure
        """
        self.layerRef = {'conv': self._createConv,
                         'fc': self._createFC,
                         'pool': self._createPool,
                         'relu': self._createRelu}

        self.network = []
        self.lastOutputSize = None
        # Data must be flattened for the FC section
        self.transitionToFC = False
        with open(structureFile) as structure:
            stuctureReader = csv.reader(structure)
            for line in stuctureReader:
                print(line)
                layerType = line[0]
                # First value is the layer type, rest are params
                numParams = len(line) - 1
                assert layerType in self.layerRef.keys()
                if self.lastOutputSize is None:
                    assert layerType == 'conv'
                    assert numParams == self.requiredParams[layerType] + 2
                else:
                    assert numParams == self.requiredParams[layerType]
                nextObject = self.layerRef[layerType](line[1:])
                self.network.append(nextObject)

    def feedThroughNetwork(self, data: np.ndarray):
        beenFlattened = False
        curMat = data
        for iLayer in self.network:
            if isinstance(iLayer, layer.FC) and not beenFlattened:
                beenFlattened = True
                curMat = np.ravel(curMat)
                iLayer.feed(curMat)
            else:
                iLayer.feed(curMat)
                curMat = iLayer.lastOutput
        return curMat

    ##########################
    ######## TRAINING ########
    ##########################

    def train(self, trainingData, testData, learningRate: float, batchSize: int, numEpochs: int, weightsFolder: str):
        """

        :param trainingData: Data object (from data.py) loaded with training data
        :param testData: Data object (from data.py) loaded with test data
        :param learningRate: Float (0,1] used for backpropgation
        :param batchSize: Number of data points in each batch
        :param numEpochs: Total number of epochs
        :param weightsFolder: Folder to save network weights in after training
        :return: training error, testing error
        """
        trainError = np.zeros(numEpochs)
        testError = np.zeros(numEpochs)

        trainFile = "results/TRAIN_E"
        testFile = "results/TEST_E"

        # Decay learning rate to avoid missing minimum
        decay = (learningRate/numEpochs)

        startTime = time()
        for epoch in range(numEpochs):
            error = 0
            trainingData.setBatches(batchSize)
            print(trainingData.numBatches)
            for batch in range(trainingData.numBatches):
                curBatch = trainingData.getBatch(batch)
                #decaying learning rate
                currentLearnRate = learningRate / (1 + decay*epoch)
                batchError = self.updateBatch(curBatch, currentLearnRate)
                error += batchError
                print("     Batch Error: ", batchError)
            trainError[epoch] = error/batchSize
            testError[epoch] = self.test(testData)

            elapsedTime = strftime("%H:%M:%S", gmtime(time() - startTime))
            print("###### Epoch {0} #########".format(epoch))
            print("Training Error = {:.5f}".format(trainError[epoch]))
            print("Validation Error = {:.5f}".format(testError[epoch]))
            print("Elapsed Time = {0}".format(elapsedTime))
            print()

            curTime = strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
            # Save the test and training error to csv files for analysis afterwords
            np.savetxt(fname=trainFile + str(epoch) + curTime + ".csv",
                       X=trainError,
                       delimiter=",")
            np.savetxt(fname=testFile + str(epoch) + curTime + ".csv",
                       X=testError,
                       delimiter=",")

        self.saveWeights(weightsFolder)
        return trainError, testError

    def test(self, data):
        testError = 0
        for image, label in data.getAll():
            predicted = self.feedThroughNetwork(image)
            error = self.calcOutputError(label, predicted)
            testError += error
        return testError

    def updateBatch(self, batchData, learningRate):
        """
        Updates the weights through backpropogation for each mini batch
        :param batchData: List of tuples (np.ndarray, class)
        :param learningRate:
        :return: Batch error
        """
        # Change in weights
        deltaWeights = []
        #Change in bias
        deltaBias = []
        # FC and CONV layers to be updated
        updatedLayers = []
        # Should move this loop to network creation so it is not called everytime
        for i in range(len(self.network)-1,-1,-1):
            curLayer = self.network[i]
            if isinstance(curLayer, layer.Conv):
                deltaWeights.append(np.zeros(curLayer.filters.shape))
                deltaBias.append(np.zeros(curLayer.numFilters))
                updatedLayers.append(i)
            elif isinstance(curLayer, layer.FC):
                deltaWeights.append(np.zeros(curLayer.nodes.shape))
                deltaBias.append(np.zeros(curLayer.numNodes))
                updatedLayers.append(i)

        totalLoss = 0
        for image, label in batchData:
            predicted = self.feedThroughNetwork(image)
            error = self.calcOutputError(label, predicted[0])
            totalLoss += error
            curDWeights = []
            curDBias = []

            # Calculate error and gradient for output first, to be backpropogated
            dErrorWeights, dOutWRTN = self.atOutput(target=label,
                                                    actual=predicted,
                                                    inputs=self.network[-1].lastInput)
            curDWeights.append(dErrorWeights[:, 0])
            curDBias.append(dOutWRTN[:, 0])
            for iLayer in range(len(self.network)-1, -1, -1):
                rightLayer = self.network[iLayer]
                if iLayer < 1:
                    leftLayer = None
                else:
                    leftLayer = self.network[iLayer - 1]

                # Layer transition type
                trans = self.backPropTransition(rightLayer, leftLayer)
                if trans == 0:
                    dErrorWeights, dOutWRTN = self.backpropFcToFc(dOutWRTN=dOutWRTN,
                                                                  priorWeights=rightLayer.nodes,
                                                                  lastOutput=leftLayer.lastOutput,
                                                                  lastInput=leftLayer.lastInput)
                    curDWeights.append(dErrorWeights)
                    # deltaBias is returned as 2d of size n x 1, we need 1d of size n
                    curDBias.append(dOutWRTN[:, 0])
                elif trans == 1:
                    # No parameters to be updated in pool
                    dOutWRTN = self.backpropFcToPool(dOutWRTN=dOutWRTN,
                                                     priorWeights=rightLayer.nodes,
                                                     poolWeights=leftLayer.weights,
                                                     poolOutSize=leftLayer.outputSize,
                                                     stride=leftLayer.stride)
                        
                elif trans == 2:
                    # From conv to relu
                    # No parameters to be updated
                    dOutWRTN = self.backpropFromConv(dOutWRTN=dOutWRTN,
                                                     convWeights=rightLayer.filters,
                                                     inputSize=rightLayer.lastInput.shape)
                    dOutWRTN = self.backprop3dToRelu(dOutWRTN=dOutWRTN,
                                                      reluWeights=leftLayer.weights)
                
                elif trans == 3:
                    # From conv to pooling
                    # No parameters to be updated
                    dOutWRTN = self.backpropFromConv(dOutWRTN=dOutWRTN,
                                                     convWeights=rightLayer.filters,
                                                     inputSize=leftLayer.outputSize,
                                                     poolWeights=leftLayer.weights,
                                                     stride=leftLayer.stride)
                        
                elif trans == 4:
                    # Pool to relu
                    # No parameters to be updated
                    dOutWRTN = self.backprop3dToRelu(dOutWRTN=dOutWRTN,
                                                     reluWeights=leftLayer.weights)
                
                elif trans == 5:
                    # First layer in the network uses the image as input data
                    # Every conv is followed by a RELU
                    if leftLayer is None:
                        convInput = image
                    else:
                        convInput = leftLayer.lastInput
                    dErrorWeights, dBias, dOutWRTN = self.backpropReluToConv(dOutWRTN=dOutWRTN,
                                                                             convOutput=leftLayer.lastOutput,
                                                                             convInput=convInput,
                                                                             filtersShape=leftLayer.filters.shape)
                    curDWeights.append(dErrorWeights)
                    curDBias.append(dBias)

            for i in range(len(curDWeights)):
                deltaWeights[i] += curDWeights[i]
                deltaBias[i] += curDBias[i]
        for iLayer, dWeight, dBias in zip(updatedLayers,deltaWeights,deltaBias):
            curLayer = self.network[iLayer]
            if isinstance(curLayer, layer.Conv):
                curLayer.filters += learningRate*dWeight
                curLayer.bias += learningRate*dBias
            else:
                curLayer.nodes += learningRate * dWeight
                curLayer.biases += learningRate * dBias
        return totalLoss


    def backPropTransition(self, rightLayer, leftLayer):
        # Fully connected to fully connected
        if isinstance(rightLayer, layer.FC) and isinstance(leftLayer, layer.FC):
            return 0 # backpropFcToFc
        # Flattened data back to final 3d pool
        elif isinstance(rightLayer, layer.FC) and isinstance(leftLayer, layer.Pool):
            return 1 # backpropFcToPool
        # From any 3d layer to relu
        elif isinstance(leftLayer, layer.Relu) and isinstance(rightLayer, layer.Conv):
            return 2 #conv to relu
        elif isinstance(leftLayer, layer.Pool) and isinstance(rightLayer, layer.Conv):
            return 3 #backpropConvTopool
        elif isinstance(rightLayer, layer.Pool) and isinstance(leftLayer, layer.Relu):
            return 4
        elif isinstance(leftLayer, layer.Conv):
            return 5 # to conv
        else:
            return 6


    #*************************************#
    #**** Network Creation and Saving*****#
    #*************************************#

    # Only used internally to create network from csv file
    def _createConv(self,paramList):
        if len(paramList) == 4:
            newConv = layer.Conv(filterSize=int(paramList[0]),
                                 numFilters=int(paramList[1]),
                                 inputSize=self.lastOutputSize,
                                 padding=int(paramList[2]),
                                 stride=int(paramList[3]))
            self.lastOutputSize = newConv.outputSize
        else:
            size = tuple([int(paramList[3]), int(paramList[3]), int(paramList[2])])
            newConv = layer.Conv(filterSize=int(paramList[0]),
                                 numFilters=int(paramList[1]),
                                 inputSize=size,
                                 padding=int(paramList[4]),
                                 stride=int(paramList[5]),
                                 first=True)
            self.lastOutputSize = newConv.outputSize
        return newConv

    def _createFC(self,paramList):
        if self.transitionToFC == False:
            # Get flattened size
            newOutputSize = 1
            for i in self.lastOutputSize:
                newOutputSize = newOutputSize * i
            self.lastOutputSize = newOutputSize
            self.transitionToFC = True
        newFC = layer.FC(numNodes=int(paramList[0]),
                         dropout=float(paramList[1]),
                         numInputs=self.lastOutputSize)
        self.lastOutputSize = newFC.numNodes
        return newFC

    def _createPool(self,paramList):
        newPool = layer.Pool(filterSize=int(paramList[0]),
                             stride=int(paramList[1]),
                             inDim = self.lastOutputSize)
        self.lastOutputSize = newPool.outputSize
        return newPool

    def _createRelu(self,paramList):
        newRelu = layer.Relu(float(paramList[0]))
        return newRelu

    def saveWeights(self, folder: str):
        """
        Saves the network weights (From convolutional and fully connected layers) and biases.
        The structure csv must be placed must be manually placed into the folder for later use
        :param folder: directory to save in
        :return:
        """
        if not self.network:
            print("Network has not been made")
        else:
            c=0
            fc = 0
            for iLayer in self.network:
                if isinstance(iLayer, layer.Conv):
                    weightName = folder + "/C" + str(c)+"weights"
                    biasName = folder + "/C" + str(c)+ "bias"
                    np.save(weightName, iLayer.filters, allow_pickle=False)
                    np.save(biasName, iLayer.bias, allow_pickle=False)
                    c += 1
                elif isinstance(iLayer, layer.FC):
                    weightName = folder + "/F" + str(fc) + "weights"
                    biasName = folder + "/F" + str(fc) + "bias"
                    np.save(weightName, iLayer.nodes, allow_pickle=False)
                    np.save(biasName, iLayer.biases, allow_pickle=False)
                    fc += 1

    def loadWeights(self, folder: str):
        """
        Loads in saved WEIGHTS only. Network structure must still be defined in a csv file
        :param folder: directory weights are saved in
        :return:
        """
        if not self.network:
            print("Network structure has not been created")
        else:
            c = 0
            fc = 0
            for iLayer in self.network:
                if isinstance(iLayer, layer.Conv):
                    weightName = folder + "/C" + str(c)+ "weights.npy"
                    biasName = folder + "/C" + str(c)+ "bias.npy"
                    iLayer.filters = np.load(weightName)
                    iLayer.bias = np.load(biasName)
                    c += 1
                elif isinstance(iLayer, layer.FC):
                    weightName = folder + "/F" + str(fc) + "weights.npy"
                    biasName = folder + "/F" + str(fc) + "bias.npy"
                    iLayer.nodes = np.load(weightName)
                    iLayer.biases = np.load(biasName)
                    fc += 1


    ##########################
    ######## BACKPROP ########
    ##########################

    def calcOutputError(self, target, output):
        """

        :param target: Actual class of data
        :param output: predicted class
        :return: LogLoss/ Binary Cross Entropy
        """
        return target*log(1/output) + (1-target)*log(1/(1-output))

    def sigmoidPrime(self, data: np.ndarray):
        return np.multiply(data, 1 - data)

    def atOutput(self, target: float, actual: float, inputs: np.ndarray):
        """
        Used to calculate delta values for output node only

        :param target: Actual class of data (1 or 0)
        :param actual: value outputted by network
        :param inputs: input array into classification node
        :return: Change in error with respect to output weights, and change of
                activation values with respect to input
        """

        # Change in error with respect to value outputted
        dErrorWRTActual = actual - target
        # change of activation values with respect to input
        dOutWRTN = dErrorWRTActual * actual*(1 - actual)
        # Reshape from (1,) to (1,1) for dot product later
        dOutWRTN = np.reshape(dOutWRTN, (1, 1))
        numInputs = inputs.shape[0]
        reshapedInputs = np.zeros((numInputs, 1))
        reshapedInputs[:, 0] = inputs
        dErrorWRTWeights = np.dot(dOutWRTN, reshapedInputs.transpose())

        return dErrorWRTWeights, dOutWRTN

    def backpropFcToFc(self, dOutWRTN: np.ndarray, priorWeights: np.ndarray, lastOutput: np.ndarray, lastInput: np.ndarray):
        """

        :param dOutWRTN: Current delta of layer activation WRT to output
        :param priorWeights: Weights leading to previously updated layer
        :param lastOutput: Output values from to be updated layer
        :param lastInput: Input values from to be updated layer
        :return: Change of error WRT to weights and layer activation WRT to output
        """
        # The derivative of the sigmoid for the  values for layer to the right
        # Updates using the weights from the right layer to following layer
        sigPrime = self.sigmoidPrime(lastOutput)
        reshapedSigPrime = np.zeros((len(sigPrime), 1))
        reshapedSigPrime[:,0] = sigPrime
        dOutWRTN = np.dot(priorWeights.transpose(), dOutWRTN) * reshapedSigPrime
        numInputs = len(lastInput)
        # Add extra dimension to allow for dot product
        lastInput = np.reshape(lastInput, (numInputs, 1))
        dErrorWRTWeights = np.dot(dOutWRTN, lastInput.transpose())

        return dErrorWRTWeights, dOutWRTN


    def backpropFcToPool(self, dOutWRTN: np.ndarray, priorWeights: np.ndarray, poolWeights: np.ndarray, poolOutSize: tuple, stride: int):
        """

        :param dOutWRTN: Backpropped delta
        :param poolWeights: Matrix representing which values were selected in pooling
        :param poolOutSize: Output size of pooling layer
        :param stride: stride of pooling filter
        :return: Upsized and unflattened change of layer activations WRT output
        """
        # No sigmoid prime used here because pool has no parameters
        dOutWRTN = np.dot(priorWeights.transpose(), dOutWRTN)
        # reshape from the flattened back to 3d
        dOutWRTN = np.reshape(dOutWRTN, poolOutSize)
        maxIndices = np.where(poolWeights > 0)
        newDOutWRTN = np.zeros(poolWeights.shape)
        for x, y, z in zip(maxIndices[0],maxIndices[1],maxIndices[2]):
            postPoolX = x // stride
            postPoolY = y // stride
            newDOutWRTN[x, y, z] = dOutWRTN[postPoolX, postPoolY, z]

        # No delta weight for pooling layer
        return newDOutWRTN

    def backprop3dToRelu(self, dOutWRTN: np.ndarray, reluWeights: np.ndarray):
        """

        :param dOutWRTN: Backpropped delta
        :param reluWeights: Indices saved in Relu layer
        :return: change of layer activations WRT output
        """
        # dx/dy a*x = a
        newDOutWRTN = np.multiply(dOutWRTN, reluWeights)
        return newDOutWRTN

    def backpropFromConv(self, dOutWRTN: np.ndarray, convWeights: np.ndarray, inputSize: tuple, poolWeights = 0, stride = 0):
        """

        :param dOutWRTN: Prior change of layer activations WRT output
        :param convWeights: Filters from previously update conv layer
        :param inputSize: Size inputted to conv layer
        :param poolWeights: If next update is to a pool layer, supply weights form Pool layer
        :param stride: Stride of pool layer
        :return: change of layer activations WRT output
        """
        numFilters = convWeights.shape[-1]
        inChannels = inputSize[-1]
        convSideSize = convWeights.shape[1]
        newDOutWRTN = np.zeros(inputSize)
        side = dOutWRTN.shape[0]
        # Yes we know this is terrible
        # Could have been done with numpy functions, however, we decided to do it
        # in a way that we completely understand the exact action happening
        for p in range(inChannels):
            for i in range(side):
                for j in range(side):
                    cur = 0
                    for q in range(numFilters):
                        for u in range(convSideSize):
                            for v in range(convSideSize):
                                cur += dOutWRTN[i - u, j - v, p] * convWeights[p, u, v, q]
                    newDOutWRTN[i, j, p] = cur

        if type(poolWeights) != int:
            upScaledNewDOutWRTN = np.zeros(poolWeights.shape)
            maxIndices = np.where(poolWeights > 1)
            for x, y, z in zip(maxIndices[0], maxIndices[1], maxIndices[2]):
                postPoolX = x // stride
                postPoolY = y // stride
                upScaledNewDOutWRTN[x, y, z] = newDOutWRTN[postPoolX, postPoolY, z]
            return upScaledNewDOutWRTN
        else:
            return newDOutWRTN



    def backpropReluToConv(self, dOutWRTN: np.ndarray, convOutput: np.ndarray, convInput, filtersShape: tuple):
        """

        :param dOutWRTN: previous change of layer activations WRT output
        :param convOutput: Output from conv layer to be updated
        :param convInput: Input to conv layer to be updated
        :param filtersShape: Shape of the filters in the conv layer to be updated
        :return: Change in error WRT conv weights, change of layer activations WRT output
        """
        sp = self.sigmoidPrime(convOutput)
        newDOutWRTN = np.multiply(sp, dOutWRTN)

        numFilters = filtersShape[0]
        filterSide = filtersShape[1]
        inputChannels = convInput.shape[-1]
        inputSide = convInput.shape[0]
        outputSide = convOutput.shape[0]
        dBias = np.zeros(numFilters)
        for layer in range(numFilters):
            dBias[layer] = newDOutWRTN[:, :, layer].sum()

        dErrorWRTWeights = np.zeros(filtersShape)

        # For each filter
        for q in range(numFilters):
            curDError = dErrorWRTWeights[q, :, :, :]
            curDOut = newDOutWRTN[:, :, q]
            # For each depth slice of each filter
            for p in range(inputChannels):
                # For each row,col index of filter
                curInput = convInput[:, :, p]
                for u in range(filterSide):
                    for v in range(filterSide):
                        curD = 0
                        for i in range(outputSide):
                            for j in range(outputSide):
                                curD += curInput[i - u, j - v] * curDOut[i, j]
                        curDError[u, v, p] = curD

        return dErrorWRTWeights, dBias, newDOutWRTN
