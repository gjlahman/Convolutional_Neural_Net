"""
Only for demo purposes. 
"""

import classifier

structureFile = 'structure.csv'

weights = 'DEC_9TH'

demo = '/Users/gjlahman/Documents/GitHub/AI_Final/demoImages'
classes = {0: "Not a hot dog", 1: "Hot dog"}

c = classifier.Classifier(structureFile, classes, weights, savedWeights=weights)
c.prepareDemo(demo)
c.waitForInput()
