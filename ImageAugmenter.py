#!/usr/bin/env python
__author__ = "Gabriel Lahman"

"""
This class uses PIL to allow images to be augmented in the following ways: Filtering, rotating, padding, and scaling.
Augmented data is used to increase the amount of training data for our CNN.
"""

from PIL import Image, ImageOps, ImageFilter
from random import choice, sample


class ImageAugmenter:
    filters = [ImageFilter.GaussianBlur,
               ImageFilter.SMOOTH,
               ImageFilter.SHARPEN,
               ImageFilter.BLUR]

    rotations = [Image.ROTATE_90,
                 Image.ROTATE_180,
                 Image.ROTATE_270,
                 Image.FLIP_LEFT_RIGHT,
                 Image.FLIP_TOP_BOTTOM,
                 Image.TRANSPOSE]

    def __init__(self, width: int, height: int):
            self.desiredWidth = width
            self.desiredHeight = height

    def scale(self, im: Image):
        """
        Scales the image inplace to desired width/height while preserving aspect ratio

        :param im: PIL Image object
        """
        w, h = im.size
        r = min(self.desiredHeight/w,
                self.desiredWidth / h)
        newW = int(r*w)
        newH = int(r*h)
        im.thumbnail((newW, newH))

    def pad(self, im: Image):
        """
        Pads image with 0's in order to fit desired width and height

        :param im: PIL Image object
        :return: PIL Image object
        """
        w, h = im.size
        horizontalPad = self.desiredWidth - w
        verticalPad = self.desiredHeight - h
        padding = (horizontalPad // 2,
                   verticalPad // 2,
                   horizontalPad - (horizontalPad // 2),
                   verticalPad - (verticalPad // 2))
        return ImageOps.expand(im, padding)

    def rotate(self, im: Image):
        """
        Rotates images using random choice from self.rotations

        :param im: PIL Image object
        :return: PIL Image object
        """
        return im.transpose(choice(self.rotations))

    def filter(self, im: Image):
        """
        Adds a random filter choice to the image from self.filters

        :param im: PIL Image object
        :return: PIL Image object
        """
        return im.filter(choice(self.filters))

    def rotateAndFilter(self, im: Image):
        """
        Helper function to rotate and filter image
        :param im: PIL image object
        :return: PIL image object
        """
        return self.rotate(self.filter(im))



