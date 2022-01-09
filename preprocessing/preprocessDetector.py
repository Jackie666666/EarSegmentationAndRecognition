import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask

def normalize3(image, mask, faceMask):
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0
    faceMask = tf.cast(faceMask, tf.float32) / 255.0
    return image, mask, faceMask

def read_image(image_path, h, w):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    return tf.image.resize(image, [h, w])

def read_mask(mask_path, h, w):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    return tf.image.resize(mask, [h, w])

def load_image(image_path, isTrain, imageH, imageW):
    input_image = read_image(image_path, imageH, imageW)

    mask_path = tf.strings.regex_replace(image_path, 'train', "/annotations/segmentation/train") if isTrain \
        else tf.strings.regex_replace(image_path, 'test', "/annotations/segmentation/test")
    
    faceMask_path = tf.strings.regex_replace(image_path, 'train', "trainFaceMask") if isTrain \
        else tf.strings.regex_replace(image_path, 'test', "testFaceMask")

    input_mask = read_mask(mask_path, imageH, imageW)
    # input_faceMask = read_mask(faceMask_path, imageH, imageW)

    input_image, input_mask = normalize(input_image, input_mask)
    # input_image, input_faceMask, input_mask = normalize3(input_image, input_faceMask, input_mask)

    # return input_image, input_mask, input_faceMask
    return input_image, input_mask

def subset_dataset(dataset, indices):
    return dataset.enumerate().filter(lambda i, t: tf.reduce_any(i == indices)).map(lambda j, u: u)

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        # self.augment_faceMask = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.HSV = [0.3, 0.3, 0.3]

    def hsv(self, image, mask):
    # def hsv(self, image, mask, faceMask):
        if tf.random.uniform([]) < self.HSV[0]:
            image = tf.image.adjust_hue(image, tf.random.uniform([], -0.1, 0.1))
        if tf.random.uniform([]) < self.HSV[1]:
            image = tf.image.adjust_saturation(image, tf.random.uniform([], 0, 2))
        if tf.random.uniform([]) < self.HSV[2]:
            image = tf.image.adjust_brightness(image, tf.random.uniform([], -0.2, 0.2))
        return image, mask
        # return image, mask, faceMask

    def call(self, inputs, labels):
    # def call(self, inputs, labels, faceMask):
        inputs = tfa.image.equalize(inputs)
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        # faceMask = self.augment_faceMask(faceMask)

        inputs, labels= self.hsv(inputs, labels)
        # inputs, labels, faceMask = self.hsv(inputs, labels, faceMask)
        # newInput = tf.concat([inputs, faceMask], axis=-1)
        return inputs, labels
        # return newInput, labels

class AugmentAdditionalMask(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.

    def call(self, inputs, labels, faceMask):
        newInput = tf.concat([inputs, faceMask], axis=-1)

        return newInput, labels