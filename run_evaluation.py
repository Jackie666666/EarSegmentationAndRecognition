import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from tensorflow import keras
import json
from tensorflow.python.ops.gen_array_ops import TensorStridedSliceUpdate
import tensorflow_addons as tfa
from detectors.my_detectors.UNet import UNet
from preprocessing.preprocess import load_image
from tqdm import tqdm
from customLoss import dice_loss

BASE_PATH = "./data"

TEST_DATA_FOLDER = pathlib.Path(BASE_PATH + "/test")
TEST_MASK_FOLDER = pathlib.Path(BASE_PATH + "/annotations/segmentation/train")
BATCH_SIZE = 8

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MODEL_NAME = "UNet-EfficientNetB0-SparseCategoricalCE"
SAVE = False
VIZ = False

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
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]
    # pred_mask = pred_mask[0]
    # pred_mask = tf.where(pred_mask>0.5,1,0)
    # return pred_mask

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
    falseNegative = np.sum(np.logical_and(np.logical_not(predMask), trueMask))
    truePositive = np.sum(np.logical_and(trueMask, predMask))
    return truePositive/(truePositive+falseNegative) if (truePositive+falseNegative)>0 else 0


from skimage.color import rgb2gray

def vizualize(maskPairs, n=3):
    fig, ax = plt.subplots(2,n, figsize=(15,15))
    for i in range(n):
        iou, imageMaskPair = maskPairs[len(maskPairs)-i-1]
        image, trueMask, predMask = imageMaskPair
        rgbMask = np.zeros(image.shape)
        predMask = np.squeeze(predMask)
        rgbMask[predMask==1] = [1,0,0]
        ax[0,i].imshow(rgb2gray(image), cmap="gray")
        ax[0,i].imshow(rgbMask, alpha=0.5)
        ax[0,i].title.set_text(f"IoU:{round(iou,3)}")
        ax[0,i].axis("off")
    
    for i in range(n):
        iou, imageMaskPair = maskPairs[i]
        image, trueMask, predMask = imageMaskPair
        rgbMask = np.zeros(image.shape)
        predMask = np.squeeze(predMask)
        rgbMask[predMask==1] = [1,0,0]
        ax[1,i].imshow(rgb2gray(image), cmap="gray")
        ax[1,i].imshow(rgbMask, alpha=0.5)
        # rgbMaskTrue = np.zeros(image.shape)
        # trueMask = np.squeeze(trueMask)
        # rgbMaskTrue[trueMask==1] = [0,1,0]
        # ax[1,i].imshow(rgbMaskTrue, alpha=0.2)
        ax[1,i].title.set_text(f"IoU:{round(iou,3)}")
        ax[1,i].axis("off")

    fig.suptitle(f"Model: {MODEL_NAME}")
    plt.tight_layout()
    plt.savefig(f"./results/{MODEL_NAME}.jpg")


if __name__ == "__main__":
    test_dataset = tf.data.Dataset.list_files(str(TEST_DATA_FOLDER/"*.png"))
    test_images = test_dataset.map(lambda x: load_image(x, False, IMAGE_HEIGHT, IMAGE_WIDTH), num_parallel_calls=tf.data.AUTOTUNE)

    # for regular loss
    # model = keras.models.load_model(f"./detectors/checkpoints/{MODEL_NAME}/weights0050.h5")

    # for SigmoidFocalCrossEntropy()
    # model = keras.models.load_model(f"./detectors/checkpoints/{MODEL_NAME}/weights0050.h5", custom_objects={"loss": tfa.losses.SigmoidFocalCrossEntropy()})
    
    # for dice loss
    # model = keras.models.load_model(f"./detectors/checkpoints/{MODEL_NAME}/weights0050.h5", compile=False)
    # model.compile(optimizer='adam',
    #         loss = dice_loss,
    #         metrics=['accuracy'])

    # Custom all (for efficientNet)
    model = UNet(IMAGE_HEIGHT, IMAGE_WIDTH, 3).get_model()
    model.load_weights(f"./detectors/checkpoints/{MODEL_NAME}/weights0030.h5")
    model.compile(optimizer="adam", loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    finalAccuracy = finalIoU = finalPrecision = finalRecall = 0
    iouMaskPairs = []

    for element in tqdm(test_images.as_numpy_iterator()):
        image, trueMask = element
        imageToPredict = image[None, :,:,:]
        predMask = model.predict(imageToPredict)
        predMask = create_mask(predMask)
       
        finalAccuracy += accuracy(trueMask, predMask)
        currentIoU = iou(trueMask, predMask)
        finalIoU += currentIoU
        finalPrecision += precision(trueMask, predMask)
        finalRecall += recall(trueMask, predMask)
        iouMaskPairs.append((currentIoU, [image, trueMask, predMask]))
    
    # for element in tqdm(test_images.as_numpy_iterator()):
    #     image, trueMask, faceMask = element
    #     imageWithMask = tf.concat([image, faceMask], axis=-1)
    #     imageToPredict = imageWithMask[None, :,:,:]
    #     predMask = model.predict(imageToPredict)
    #     predMask = create_mask(predMask)
       
    #     finalAccuracy += accuracy(trueMask, predMask)
    #     currentIoU = iou(trueMask, predMask)
    #     finalIoU += currentIoU
    #     finalPrecision += precision(trueMask, predMask)
    #     finalRecall += recall(trueMask, predMask)
    #     iouMaskPairs.append((currentIoU, [image, trueMask, predMask]))


    finalAccuracy /= test_images.cardinality().numpy()
    finalIoU /= test_images.cardinality().numpy()
    finalPrecision /= test_images.cardinality().numpy()
    finalRecall /= test_images.cardinality().numpy()
    finalF1 = (2*finalPrecision*finalRecall)/(finalPrecision+finalRecall) if (finalPrecision+finalRecall) > 0 else 0
    outString = f"{MODEL_NAME}\nAcc: {finalAccuracy}\nIoU: {finalIoU}\nPrecision: {finalPrecision}\nRecall: {finalRecall}\nF1: {finalF1}"
    print(outString)
    if SAVE:
        with open(f"./results/{MODEL_NAME}.txt", "w+") as outFile:
            outFile.write(outString)

    if VIZ:
        iouMaskPairs.sort(key=lambda x:x[0])
        vizualize(iouMaskPairs, n=5)