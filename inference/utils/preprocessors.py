#  Copyright (c) 2019 by LatentAI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "LatentAI Commercial Software License".
#  Please see the LICENSE file that should have been included as part of
#  this package.
#
# @file    preprocessors.py
#
# @author  Videet Parekh
#
# @date    Wed, 16 Dec 20

from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import sys
import os

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


class ImagePreprocessor():
    def none(self, img):
        return img

    def bgrtorgbcaffe(self, img):
        img = img[:, :, :, ::-1]  # RGB to BGR
        return preprocess_input(img, mode='caffe', data_format='channels_last')

    def bgrtorgb(self, img):
        return img[:, :, :, ::-1]  # RGB to BGR

    def bgrtorgb2(self, img):
        img = img[:, :, :, ::-1]  # RGB to BGR
        return preprocess_input(img)

    def bgrtorgb3(self, img):
        img = img[:, :, :, ::-1]  # RGB to BGR
        return preprocess_input(img, mode='tf')

    def imagenet(self, img):
        return preprocess_input(img, mode='tf')

    def imagenet_caffe(self, img):
        return preprocess_input(img, mode='caffe')

    def symm(self, img):
        return np.array(np.array(img)-128, dtype=np.int8)

    def float32(self, img):
        return img.astype(np.float32)

    def uint8(self, img):
        return img.astype(np.uint8)

    def rgbtogray(self, img):
        if len(img.shape) == 4:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img = np.expand_dims(img, axis=-1)
        return img.astype(np.float32)

    def rgbtogray_int8(self, img):
        if len(img.shape) == 4:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img = np.expand_dims(img, axis=-1)
        return img.astype(np.uint8)
