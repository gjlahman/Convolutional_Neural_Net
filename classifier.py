import signal
import sys
import numpy as np
from PIL import Image
from random import randint

import network
import data
import ImageAugmenter


class Classifier:
    def __init__(self, structureFile: str, classLabels, weightFldr: str, savedWeights = None):
        """

        :param structureFile: File defining network structure
        :param classLabels: Dictionary with int key values and class values
        :param weightFldr: Folder to save weights in
        :param savedWeights: Only used if you're loading an existing network
        """
        self.network = network.Network(structureFile)
        if savedWeights is not None:
            self.network.loadWeights(savedWeights)
        self.classes = classLabels
        self.weightFldr = weightFldr
        self.size = (200,200)
        self.augmenter = ImageAugmenter.ImageAugmenter(self.size[0], self.size[1])
        self.demoPath = None

    def classify(self, imagePath: str):
        """

        :param imagePath: Path to image to classify
        :return:
        """
        im = Image.open(imagePath)
        im.show()
        self.augmenter.scale(im)
        im = self.augmenter.pad(im)
        im = np.asarray(im, dtype=float)
        #Scale from -.5 to .5
        im = (im / 255) - .5
        print(im.shape)
        prediction = self.network.feedThroughNetwork(im)[0]
        print("Percent : {0}".format(prediction))
        prediction = (prediction > .5) * 1
        return self.classes[prediction]

    def trainNetwork(self,trainingData, testData, learningRate, batchSize, numEpochs):
        """

        :param trainingData: data.Data object
        :param testData: data.Data object
        :param learningRate: float (0,1]
        :param batchSize: Number of images per batch
        :param numEpochs: Epochs to train for
        :return: training error and testing error
        """
        trainError, testError = self.network.train(trainingData,
                                                   testData,
                                                   learningRate,
                                                   batchSize,
                                                   numEpochs,
                                                   self.weightFldr)

    def prepareDemo(self, pathToDemo):
        """

        :param pathToDemo: Folder containing demo images
        :return:
        """
        self.demoPath = pathToDemo

    def waitForInput(self):
        """
        Interactive demo
        :return:
        """
        while True:
            file = input("Enter the object you want to test: ")
            if file == "done":
                break
            elif file == "hotdog":
                    file += str(randint(1,7))
            file = self.demoPath + "/" + file + ".jpg"
            try:
                p = self.classify(file)
                print(p)
            except FileNotFoundError:
                print("Not available in demo mode")
        
        
        
    def saveNetwork(self):
        self.network.saveWeights(self.weightFldr)


if __name__ == "__main__":
    structureFile = "structure.csv"
    weightsFolder = "DEC_9TH"
    classes = {0: "Not a hot dog", 1: "Hot dog"}
    classifier = Classifier(structureFile, classes, weightsFolder)

    #Signal handler allows the training to be cancelled with CTRL-C, and saves the network weights
    def signal_handler(signal, frame):
        print("SAVING NETWORK")
        classifier.saveNetwork()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    trainPositive = "/Users/gjlahman/Documents/GitHub/AI_Final/img2/train/hotDog"
    trainNeg = "/Users/gjlahman/Documents/GitHub/AI_Final/img2/train/notHotDog"
    testPositive = "/Users/gjlahman/Documents/GitHub/AI_Final/img2/test/hotDog"
    testNeg = "/Users/gjlahman/Documents/GitHub/AI_Final/img2/test/notHotDog"

    trainingData = data.Data(trainPositive, trainNeg)
    testData = data.Data(testPositive, testNeg)

    epochs = 25
    learningRate = .25
    batchSize = 16
    classifier.trainNetwork(trainingData, testData, learningRate, batchSize, epochs)
    
    



