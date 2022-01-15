import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import pathlib
import csv
import random
from preprocessing.preprocessClassifier import load_image, Augment
from feature_extractors.my_extractors.transferLearning import VGG16, ResNet50, ResNet101, DenseNet121, EfficientNetB0
from feature_extractors.my_extractors.customNet import CustomNet

BASE_PATH = "./data/croppedEars"
TRAIN_DATA_FOLDER = pathlib.Path(BASE_PATH + "/train")
ANOTATIONS_PATH = "./data/annotations/recognition/ids.csv"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
VAL_RATIO = 0.15

BATCH_SIZE = 16
EPOCHS = 50

MODEL_NAME = "ResNet101-50E-myEars"

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

def display(image, label):
    randomInt = random.randrange(100)
    plt.figure(randomInt, figsize=(13,13))
    plt.axis('off')
    plt.imshow(image.numpy())
    plt.title(label.numpy(), fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(f"./feature_extractors/checkpoints/{MODEL_NAME}"):
        os.makedirs(f"./feature_extractors/checkpoints/{MODEL_NAME}")
    if not os.path.exists(f"./feature_extractors/figures/{MODEL_NAME}"):
        os.makedirs(f"./feature_extractors/figures/{MODEL_NAME}")

    filenames, labels = filenamesAndLabels(TRAIN_DATA_FOLDER)

    filenames_train, filenames_val, labels_train, labels_val = train_test_split(filenames, labels, test_size=VAL_RATIO, 
                                    random_state=42, shuffle=True, stratify=labels)

    dataset_train = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
    train_images = dataset_train.map(lambda x, y: load_image(x, y, IMAGE_HEIGHT, IMAGE_WIDTH), num_parallel_calls=tf.data.AUTOTUNE)

    dataset_val = tf.data.Dataset.from_tensor_slices((filenames_val, labels_val))
    val_images = dataset_val.map(lambda x, y: load_image(x, y, IMAGE_HEIGHT, IMAGE_WIDTH), num_parallel_calls=tf.data.AUTOTUNE)

    TRAIN_LENGTH = len(filenames_train)
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    train_batches = (
        train_images
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_batches = val_images.batch(BATCH_SIZE)

    # for images, labels in train_batches.take(100):
    #     sample_image, sample_label = images[0], labels[0]
    #     display(sample_image, sample_label)
    #     break
    # for images, labels in val_batches.take(1):
    #     sample_image, sample_label = images[0], labels[0]
    #     display(sample_image, sample_label)
    #     break
        
    # model = CustomNet(IMAGE_HEIGHT, IMAGE_WIDTH, 3).build_model()
    model = ResNet101(IMAGE_HEIGHT, IMAGE_WIDTH, 3).build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    model.summary()

    checkPoint_callback = tf.keras.callbacks.ModelCheckpoint("./feature_extractors/checkpoints/"+MODEL_NAME+"/weights{epoch:04d}.h5",
                                    save_weights_only=False, period=10)

    model_history = model.fit(train_batches, epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=val_batches, callbacks=[checkPoint_callback])

    history_dict = model_history.history
    json.dump(history_dict, open(f"./feature_extractors/checkpoints/{MODEL_NAME}/modelHistory.json", 'w'))

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./feature_extractors/figures/{MODEL_NAME}/loss.jpg")

    # # print("FINE-TUNING")
    # model = tf.keras.models.load_model(f"./feature_extractors/checkpoints/{MODEL_NAME}/weights0050.h5")
    
    # for layer in model.layers:
    #     if isinstance(layer, tf.keras.layers.BatchNormalization):
    #         layer.trainable = False
    #     else:
    #         layer.trainable = True
    
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #         metrics=['accuracy'])
    # model.summary()
    
    # model_history = model.fit(train_batches, epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,
    #         validation_data=val_batches, 
    #         callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    #     )
    # model.save("./feature_extractors/checkpoints/"+MODEL_NAME+"/weightsLast.h5")

    # history_dict = model_history.history
    # json.dump(history_dict, open(f"./feature_extractors/checkpoints/{MODEL_NAME}/modelHistory2.json", 'w'))

    # loss = model_history.history['loss']
    # val_loss = model_history.history['val_loss']

    # plt.figure()
    # plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    # plt.plot(model_history.epoch, val_loss, 'b', label='Validation loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss Value')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"./feature_extractors/checkpoints/{MODEL_NAME}/loss2.jpg")