import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from tensorflow import keras
import json
import tensorflow_addons as tfa
from detectors.my_detectors.UNet import UNet
from preprocessing.preprocessDetector import load_image
from tqdm import tqdm
from customLoss import dice_loss
import cv2
from PIL import Image

BASE_PATH = "./data"

TRAIN_DATA_FOLDER = pathlib.Path(BASE_PATH + "/train")

TEST_DATA_FOLDER = pathlib.Path(BASE_PATH + "/test")

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MODEL_NAME = "UNet-EfficientNetB0-SparseCategoricalCE"

TRAIN_SAVE_FOLDER = pathlib.Path(BASE_PATH + "/croppedEars/myTrain")
TEST_SAVE_FOLDER = pathlib.Path(BASE_PATH + "/croppedEars/myTest")


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def extractEar(mask, image):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    
    mask = mask.numpy().astype(np.uint8)*255

    keypoints = sorted(detector.detect(mask), reverse=True, key = lambda x: x.size)
    if len(keypoints)>=1:
        keypoint = keypoints[0]
        x1 = round(keypoint.pt[1]-keypoint.size)
        y1 = round(keypoint.pt[0]-keypoint.size)
        x2 = round(keypoint.pt[1]+keypoint.size)
        y2 = round(keypoint.pt[0]+keypoint.size)
        return image[x1:x2,y1:y2,:]
    else:
        return None


if __name__ == "__main__":
    train_filenames = os.listdir(TRAIN_DATA_FOLDER)
    train_paths = [os.path.join(TRAIN_DATA_FOLDER,x) for x in train_filenames]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    train_images = train_dataset.map(lambda x: load_image(x, True, IMAGE_HEIGHT, IMAGE_WIDTH), num_parallel_calls=tf.data.AUTOTUNE)

    # Custom all (for efficientNet)
    model = UNet(IMAGE_HEIGHT, IMAGE_WIDTH, 3).get_model()
    model.load_weights(f"./detectors/checkpoints/{MODEL_NAME}/weights0030.h5")
    model.compile(optimizer="adam", loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    for i, element in enumerate(train_images.as_numpy_iterator()):
        image, trueMask = element
        imageToPredict = image[None, :,:,:]
        predMask = model.predict(imageToPredict)
        predMask = create_mask(predMask)
        ear = extractEar(predMask, image)
        if ear is None or 0 in ear.shape:
            print(f"No ear detected here: train/{train_filenames[i]}")
        else:
            savePath = os.path.join(TRAIN_SAVE_FOLDER, train_filenames[i])
            plt.imsave(savePath, ear)

    test_filenames = os.listdir(TEST_DATA_FOLDER)
    test_paths = [os.path.join(TEST_DATA_FOLDER,x) for x in test_filenames]
    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_images = test_dataset.map(lambda x: load_image(x, False, IMAGE_HEIGHT, IMAGE_WIDTH), num_parallel_calls=tf.data.AUTOTUNE)

    for i, element in enumerate(test_images.as_numpy_iterator()):
        image, trueMask = element
        imageToPredict = image[None, :,:,:]
        predMask = model.predict(imageToPredict)
        predMask = create_mask(predMask)
        ear = extractEar(predMask, image)
        if ear is None or 0 in ear.shape:
            print(f"No ear detected here: test/{test_filenames[i]}")
        else:
            savePath = os.path.join(TEST_SAVE_FOLDER, test_filenames[i])
            plt.imsave(savePath, ear)