import numpy as np
from keras.models import load_model
from PIL import Image
import ImageAugmenter
import os
from random import randint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class trainedPredictor:
    def __init__(self, modelFile, imageDirectory):
        self.size = (200,200)
        self.imageDirectory = imageDirectory
        self.model = load_model(modelFile)
        self.augmenter = ImageAugmenter.ImageAugmenter(self.size[0], self.size[1])
        self.classes = {0: "Not a hot dog!", 1:"It's a hot dog!"}

    def predictClass(self, imageName):
        curIm = Image.open(self.imageDirectory + "/" + imageName)
        curIm.show()
        self.augmenter.scale(curIm)
        curIm = self.augmenter.pad(curIm)
        imAsArr = np.asarray(curIm, dtype=float)
        imAsArr = imAsArr / 255
        imAsArr = np.expand_dims(imAsArr, axis=0)
        prediction = self.model.predict(imAsArr, verbose=1)
        print("Percent : {0}".format(prediction[0,0]))
        whichClass = (prediction[0][0] > .5) * 1
        return self.classes[whichClass]

    def waitForInput(self):
        while True:
            file = input("Enter the object you want to test: ")
            if file == "hotdog":
                    file += str(randint(1,7))
            elif file == "done":
                break
            file += '.jpg'
            try:
                p = self.predictClass(file)
                print(p)
                print()
            except FileNotFoundError:
                print("Not available in demo mode")
                print()
        
demo = '/Users/gjlahman/Documents/GitHub/AI_Final/demoImages'
model = 'hotdogNetwork.h5'


p = trainedPredictor(model, demo)
p.waitForInput()
