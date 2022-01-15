import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_PATH = "./resultsClassifier"
FOR_MY_EARS = False
FOR_FT = True
SAVE = False
SAVE_NAME = "ResNet101-myEars-FTimportance"
if __name__ == "__main__":
    files = [x for x in os.listdir(RESULTS_PATH) if x[-4:] == ".txt"]
    
    data = []
    modelNames = []
    for currFile in files:
        currData = np.loadtxt(os.path.join(RESULTS_PATH, currFile))
        temp = currFile.split("-")

        # if "ResNet101" in temp[0]:
        #     if "myEars" in currFile:
        #         data.append(currData)
        #         modelNames.append("ResNet101-FineTunning" if "FT" in currFile else "ResNet101")

        # if temp[0] == "VGG16":
        #     continue

        if FOR_MY_EARS and "myEars" in currFile:
            if FOR_FT and "FT" in currFile:
                print(currFile)
                data.append(currData)
                modelNames.append(temp[0])
            elif not FOR_FT and "FT" not in currFile:
                print(currFile)
                data.append(currData)
                modelNames.append(temp[0])
        elif not FOR_MY_EARS and "myEars" not in currFile:
            if FOR_FT and "FT" in currFile:
                print(currFile)
                data.append(currData)
                modelNames.append(temp[0])
            elif not FOR_FT and "FT" not in currFile:
                print(currFile)
                modelNames.append(temp[0])
                data.append(currData)
    
    plt.figure(figsize=(10,10))
    for i, model in enumerate(modelNames):
        currRanks = data[i]
        plt.plot(np.arange(1,len(currRanks)+1), currRanks,label=model)

    plt.title('Cumulative Match Curve')
    plt.xlabel('Rank')
    plt.ylabel('Probability of identification')
    plt.legend()
    plt.savefig(f"./resultsClassifier/CMC-{SAVE_NAME}.jpg") if SAVE else plt.show()

