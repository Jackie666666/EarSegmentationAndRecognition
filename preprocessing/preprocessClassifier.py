import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2

def normalize(input_image, label):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, label

def read_image(image_path, h, w):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    return tf.image.resize(image, [h, w])

def load_image(image_path, label, imageH, imageW):
    input_image = read_image(image_path, imageH, imageW)
    input_image, label = normalize(input_image, label)
    return input_image, label

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.HSV = [0.3, 0.3, 0.3]

    def hsv(self, image):
        if tf.random.uniform([]) < self.HSV[0]:
            image = tf.image.adjust_hue(image, tf.random.uniform([], -0.1, 0.1))
        if tf.random.uniform([]) < self.HSV[1]:
            image = tf.image.adjust_saturation(image, tf.random.uniform([], 0, 2))
        if tf.random.uniform([]) < self.HSV[2]:
            image = tf.image.adjust_brightness(image, tf.random.uniform([], -0.2, 0.2))
        return image

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        inputs = self.hsv(inputs)

        return inputs, labels