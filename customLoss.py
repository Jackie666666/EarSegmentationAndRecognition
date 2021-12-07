import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import binary_crossentropy

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred = K.cast(y_pred, 'float32')
#     y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
#     intersection = y_true_f * y_pred_f
#     score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
#     return score

# def dice_loss(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = y_true_f * y_pred_f
#     score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return 1. - score

# def bce_dice_loss(y_true, y_pred):
#     return tf.loses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# def bce_logdice_loss(y_true, y_pred):
#     return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))