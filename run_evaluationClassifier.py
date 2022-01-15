import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import pathlib
import csv
import random
from preprocessing.preprocessClassifier import load_image
from tqdm import tqdm

BASE_PATH = "./data/croppedEars"
TEST_DATA_FOLDER = pathlib.Path(BASE_PATH + "/test")
ANOTATIONS_PATH = "./data/annotations/recognition/ids.csv"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MODEL_NAME = "ResNet101-50E-myEars-NoAug"

def filenamesAndLabels(path, train=True):
    filenames = ["train/"+x for x in os.listdir(path)] if train else ["test/"+x for x in os.listdir(path)]
    annotationsDict = {}
    
    with open(ANOTATIONS_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            annotationsDict[row[0]] = int(row[1]) - 1
    
    labels = [annotationsDict[x] for x in filenames]
    # filenames = [os.path.join(BASE_PATH, x) for x in filenames]
    filenames = [os.path.join(BASE_PATH, "myT"+x[1:]) for x in filenames] # to use my dataset
    return filenames, labels

def calcRankN(labelsProbs, trueLabel, n=1):
    if n >= len(labelsProbs):
        return None
    sortedProbs = np.argsort(labelsProbs)[::-1]
    if trueLabel in sortedProbs[:n]:
        return True
    else:
        return False

def displayCMC(ranks, save=False, saveName="CMC"):
    plt.plot(np.arange(1,len(ranks)+1), ranks)
    plt.title('Cumulative Match Curve')
    plt.suptitle(f"{MODEL_NAME}")
    plt.xlabel('Rank')
    plt.ylabel('Probability of identification')
    plt.savefig(f"./resultsClassifier/{MODEL_NAME}-{saveName}.jpg") if save else plt.show()
    if save:
        np.savetxt(f"./resultsClassifier/{MODEL_NAME}-{saveName}-ranks.txt", ranks)


if __name__ == "__main__":
    filenames, labels = filenamesAndLabels(TEST_DATA_FOLDER, train=False)

    dataset_test = tf.data.Dataset.from_tensor_slices((filenames, labels))
    test_images = dataset_test.map(lambda x, y: load_image(x, y, IMAGE_HEIGHT, IMAGE_WIDTH), num_parallel_calls=tf.data.AUTOTUNE)
    
    model = tf.keras.models.load_model(f"./feature_extractors/checkpoints/{MODEL_NAME}/weights0050.h5")
    # For transfer learning
    # model = tf.keras.models.load_model(f"./feature_extractors/checkpoints/{MODEL_NAME}/weightsLast.h5")

    ranks = np.zeros(99)
    for element in tqdm(test_images.as_numpy_iterator()):
        image, label = element
        imageToPredict = image[None, :,:,:]
        labelsProbs = model.predict(imageToPredict)[0]
        for i in range(1, 100):
            ranks[i-1] += calcRankN(labelsProbs, label, n=i)

    ranks /= len(filenames)
    displayCMC(ranks, save=True, saveName="CMC")