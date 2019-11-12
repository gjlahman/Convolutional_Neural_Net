#!/usr/bin/env python
__author__ = "Gabriel Lahman"

"""
Implements the ImageAugmenter class in order to augment all images contained within a specific directory.
"""

import os
import ImageAugmenter
from PIL import Image


class DatasetAugmenter:

    # TODO add inplace option

    def __init__(self, inDir: str, outDir: str, size: tuple):
        """
        :param inDir: Absolute path to folder containing images
        :param outDir: Name of folder to place augmented images in (Created in
                        parent directory of inDir)
        :param size: width, height to scale images to, ints
        """
        self.inDir = inDir
        os.chdir(self.inDir)
        os.chdir("..")
        os.mkdir(outDir)
        self.outDir = os.getcwd() + "/" + outDir + "/"
        os.chdir(inDir)
        self.augmenter = ImageAugmenter.ImageAugmenter(size[0],size[1])5
        self.edits = [self.augmenter.rotate,
                      self.augmenter.filter,
                      self.augmenter.rotateAndFilter]

    def augmentDataset(self):
        # Assuming every file in the directory is an image
        for imageFile in os.listdir():
            imagePath = self.inDir + "/" + imageFile
            splitFile = imageFile.split(".")
            outName = splitFile[0] + "_"
            outExt = splitFile[1]
            outPath = self.outDir + outName
            im = Image.open(imagePath).copy()
            if im.mode != "RGB":
                im = im.convert('RGB')
            edited = [im]
            for edit in self.edits:
                edited.append(edit(im))

            for i in range(len(edited)):
                self.augmenter.scale(edited[i])
                edited[i] = self.augmenter.pad(edited[i])

                edited[i].save(outPath + str(i) + '.' + outExt,
                               "JPEG")

"""
i = "/Users/gjlahman/Desktop/hotdogs_dataset/hot_dog"
o = "test"
size = (227,227)
d = DatasetAugmenter(i,o,size)
"""

