import os
from shutil import copyfile

BASE_DIR = "./data/croppedEars"

if __name__ == "__main__":
    with open("missingEars.txt") as handle:
        for line in handle:
            temp = line.rstrip().split(":")[1]
            filename = temp.strip()
            temp = filename.split("/")
            if temp[0] == "train":
                fromPath = os.path.join(BASE_DIR, filename)
                toPath = os.path.join(BASE_DIR, "myTrain", temp[1])
                copyfile(fromPath, toPath)
            else:
                fromPath = os.path.join(BASE_DIR, filename)
                toPath = os.path.join(BASE_DIR, "myTest", temp[1])
                copyfile(fromPath, toPath)