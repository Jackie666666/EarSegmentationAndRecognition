import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pathlib
import csv
from preprocessing.preprocessClassifier import load_image, Augment
from tqdm import tqdm
from sklearn.svm import SVC

BASE_PATH = "./data/croppedEars"
TRAIN_DATA_FOLDER = pathlib.Path(BASE_PATH + "/train")
TEST_DATA_FOLDER = pathlib.Path(BASE_PATH + "/test")
ANOTATIONS_PATH = "./data/annotations/recognition/ids.csv"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MODEL_NAME = "EfficientNetB0-50E-myEars"

def filenamesAndLabels(path, train=True):
    filenames = ["train/"+x for x in os.listdir(path)] if train else ["test/"+x for x in os.listdir(path)]
    annotationsDict = {}
    
    with open(ANOTATIONS_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            annotationsDict[row[0]] = int(row[1]) - 1
    
    labels = [annotationsDict[x] for x in filenames]
    # filenames = [os.path.join(BASE_PATH, x) for x in filenames]
    filenames = [os.path.join(BASE_PATH, "myT"+x[1:]) for x in filenames] # to use myEars

    return filenames, labels

def evalSVM(X_train, X_test, y_train, kernel="rbf",c=1):
    svm = SVC(C=c, kernel=kernel, random_state=42, probability=True)
    model = svm.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    return y_pred

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
    plt.xlabel('Rank')
    plt.ylabel('Probability of identification')
    plt.savefig(f"./resultsClassifier/{MODEL_NAME}-{saveName}.jpg") if save else plt.show()
    if save:
        np.savetxt(f"./resultsClassifier/{MODEL_NAME}-{saveName}-ranks.txt", ranks)

if __name__ == "__main__":
    filenames_train, labels_train = filenamesAndLabels(TRAIN_DATA_FOLDER)
    filenames_test, labels_test = filenamesAndLabels(TEST_DATA_FOLDER, train=False)

    dataset_train = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
    train_images = dataset_train.map(lambda x, y: load_image(x, y, IMAGE_HEIGHT, IMAGE_WIDTH), num_parallel_calls=tf.data.AUTOTUNE)

    dataset_test = tf.data.Dataset.from_tensor_slices((filenames_test, labels_test))
    test_images = dataset_test.map(lambda x, y: load_image(x, y, IMAGE_HEIGHT, IMAGE_WIDTH), num_parallel_calls=tf.data.AUTOTUNE)
    
    # model = tf.keras.models.load_model(f"./feature_extractors/checkpoints/{MODEL_NAME}/weights0050.h5")
    # For transfer learning
    model = tf.keras.models.load_model(f"./feature_extractors/checkpoints/{MODEL_NAME}/weightsLast.h5")

    layer_name = "flatten"
    intermidiate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    X_train = []
    y_train = np.array(labels_train)
    for element in tqdm(train_images.as_numpy_iterator()):
        image, label = element
        imageToPredict = image[None, :,:,:]
        labelsProbs = intermidiate_layer_model.predict(imageToPredict)[0]
        X_train.append(labelsProbs)
    X_train = np.array(X_train)

    X_test = []
    y_test = np.array(labels_test)
    for element in tqdm(test_images.as_numpy_iterator()):
        image, label = element
        imageToPredict = image[None, :,:,:]
        labelsProbs = intermidiate_layer_model.predict(imageToPredict)[0]
        X_test.append(labelsProbs)
    X_test = np.array(X_test)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_pred = evalSVM(X_train, X_test, y_train)

    ranks = np.zeros(99)
    for i in tqdm(range(y_pred.shape[0])):
        label = y_test[i]
        labelsProbs = y_pred[i,:]
        for i in range(1, 100):
            ranks[i-1] += calcRankN(labelsProbs, label, n=i)

    ranks /= len(filenames_test)
    displayCMC(ranks, save=True, saveName="FT+SVM")
