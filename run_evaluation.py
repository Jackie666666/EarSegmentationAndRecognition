import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from tensorflow import keras
import json
import tensorflow_addons as tfa
from preprocessing.preprocess import load_image
from tqdm import tqdm
from customLoss import dice_loss

BASE_PATH = "./data"

TEST_DATA_FOLDER = pathlib.Path(BASE_PATH + "/test")
TEST_MASK_FOLDER = pathlib.Path(BASE_PATH + "/annotations/segmentation/train")
BATCH_SIZE = 8

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MODEL_NAME = "DepLabV3+-ResNet50-BinaryCrossEntropy"

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_mask(pred_mask):
    # pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask = pred_mask[..., tf.newaxis]
    # return pred_mask[0]
    pred_mask = pred_mask[0]
    pred_mask = tf.where(pred_mask>0.5,1,0)
    return pred_mask

def accuracy(trueMask, predMask):
    tp = np.sum(np.logical_and(trueMask, predMask))
    tn = np.sum((trueMask==0)&(predMask==0))
    p = np.sum(trueMask == 1)
    n = np.sum(trueMask == 0)
    return (tp+tn)/(p+n)

def iou(trueMask, predMask):
        intersection = np.logical_and(trueMask, predMask)
        union = np.logical_or(trueMask, predMask)
        return np.sum(intersection) / np.sum(union)

def precision(trueMask, predMask):
    predictedPositive = np.sum(predMask == 1)
    truePositive = np.sum(np.logical_and(trueMask, predMask))
    return truePositive/predictedPositive if predictedPositive>0 else 0

def recall(trueMask, predMask):
    actualPositive = np.sum(trueMask == 1)
    truePositive = np.sum(np.logical_and(trueMask, predMask))
    return truePositive/actualPositive if actualPositive>0 else 0

# def recall(gt, p):
#     TP = np.sum(np.logical_and(p, gt))
#     FN = np.sum(np.logical_and(np.logical_not(p), gt))
#     try:
#         if (TP+FN) > 0:
#             return TP/(TP+FN)
#         else:
#             return 0
#     except:
#         return 0


if __name__ == "__main__":
    test_dataset = tf.data.Dataset.list_files(str(TEST_DATA_FOLDER/"*.png"))
    test_images = test_dataset.map(lambda x: load_image(x, False, IMAGE_HEIGHT, IMAGE_WIDTH), num_parallel_calls=tf.data.AUTOTUNE)

    # for regular loss
    model = keras.models.load_model(f"./detectors/checkpoints/{MODEL_NAME}/weights0050.h5")

    # for SigmoidFocalCrossEntropy()
    # model = keras.models.load_model(f"./detectors/checkpoints/{MODEL_NAME}/weights0050.h5", custom_objects={"loss": tfa.losses.SigmoidFocalCrossEntropy()})
    
    # for dice loss
    # model = keras.models.load_model(f"./detectors/checkpoints/{MODEL_NAME}/weights0050.h5", compile=False)
    # model.compile(optimizer='adam',
    #         loss = dice_loss,
    #         metrics=['accuracy'])

    finalAccuracy = finalIoU = finalPrecision = finalRecall = 0
    for element in tqdm(test_images.as_numpy_iterator()):
        image, trueMask = element
        imageToPredict = image[None, :,:,:]
        predMask = model.predict(imageToPredict)
        predMask = create_mask(predMask)
        finalAccuracy += accuracy(trueMask, predMask)
        finalIoU += iou(trueMask, predMask)
        finalPrecision += precision(trueMask, predMask)
        finalRecall += recall(trueMask, predMask)

    finalAccuracy /= test_images.cardinality().numpy()
    finalIoU /= test_images.cardinality().numpy()
    finalPrecision /= test_images.cardinality().numpy()
    finalRecall /= test_images.cardinality().numpy()
    outString = f"{MODEL_NAME}\nAcc: {finalAccuracy}\nIoU: {finalIoU}\nPrecision: {finalPrecision}\nRecall: {finalRecall}"
    print(outString)
    with open(f"./results/{MODEL_NAME}.txt", "w+") as outFile:
        outFile.write(outString)