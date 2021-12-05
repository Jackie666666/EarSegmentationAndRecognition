import os
import random

PATH = "./data/ears"

def getTestTxt():
    paths = []
    testPath = os.path.join(PATH, "test")
    for image in os.listdir(testPath):
        currPath = os.path.join(PATH, "test", image)
        paths.append(currPath)
    return paths

def getTrainValTxt():
    paths = []
    trainPath = os.path.join(PATH, "train")
    trainSize = 0.7
    for image in os.listdir(trainPath):
        currPath = os.path.join(PATH, "train", image)
        paths.append(currPath)
    random.Random(42).shuffle(paths)
    trainPaths = paths[0:int(len(paths)*trainSize)]
    valPaths = paths[int(len(paths)*trainSize):]
    return trainPaths, valPaths

def saveTxt(data, name):
    textfile = open(f"{PATH}/{name}.txt", "w")
    for element in data:
        textfile.write(element + "\n")
    textfile.close()


if __name__ == "__main__":
    testTxt = getTestTxt()
    saveTxt(testTxt, "test")
    trainTxt, valTxt = getTrainValTxt()
    saveTxt(trainTxt, "train")
    saveTxt(valTxt, "val")
