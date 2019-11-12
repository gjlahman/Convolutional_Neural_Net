import os
import numpy as np
from PIL import Image
from random import shuffle

class Data:
    def __init__(self, pathToPositive: str, pathToNegative: str):
        """

        :param pathToPositive: Full Folder containing positive classification images
        :param pathToNegative: Full Folder containing negative classification images
        """
        self.imagesDict = {}
        self.allImageNames = []
        imageNames = os.listdir(pathToPositive)
        for name in imageNames:
            fullPath = pathToPositive + "/" + name
            self.imagesDict[fullPath] = 1
            self.allImageNames.append(fullPath)
        imageNames = os.listdir(pathToNegative)
        for name in imageNames:
            fullPath = pathToNegative + "/" + name
            self.imagesDict[fullPath] = 0
            self.allImageNames.append(fullPath)
        self.batches = None
        self.numBatches = None

    def setBatches(self, batchSize):
        """
        Shuffles and partitions data into batches
        :param batchSize: Size of batches
        :return:
        """
        self.batches = []
        shuffle(self.allImageNames)
        self.numBatches = len(self.allImageNames) // batchSize
        for batch in range(self.numBatches):
            curBatch = self.allImageNames[batch*batchSize:batch*batchSize + batchSize]
            self.batches.append(curBatch)

    def getBatch(self, batchNum):
        """
        Loads images and data into memory
        :param batchNum: Which batch
        :return: List of (image data, class)
        """
        nextBatch = []
        names = self.batches[batchNum]
        for name in names:
            curImg = Image.open(name)
            curImgArray = np.asarray(curImg, dtype=float)
            #Scale image between -0.5 and 0.5
            curImgArray = (curImgArray/255) - .5
            imgClass = self.imagesDict[name]
            nextBatch.append(tuple([curImgArray, imgClass]))
        return nextBatch

    def getAll(self):
        """

        :return: All images and classes
        """
        all = []
        for name in self.allImageNames:
            curImg = Image.open(name)
            curImgArray = np.asarray(curImg, dtype=float)
            #Scale image between -0.5 and 0.5
            curImgArray = (curImgArray/255) - .5
            imgClass = self.imagesDict[name]
            all.append(tuple([curImgArray,imgClass]))
        return all

    def __len__(self):
        return len(self.allImageNames)




