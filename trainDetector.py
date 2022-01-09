import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from tensorflow import keras
from tensorflow._api.v2 import image
import tensorflow_addons as tfa
import json
from preprocessing.preprocessDetector import  Augment, AugmentAdditionalMask, subset_dataset, load_image
from detectors.my_detectors.UNet import UNet
from detectors.my_detectors.DeepLabV3 import DeepLabV3
from customLoss import dice_loss

BASE_PATH = "./data"

TRAIN_DATA_FOLDER = pathlib.Path(BASE_PATH + "/train")

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 4

VAL_RATIO = 0.85
BATCH_SIZE = 8
EPOCHS = 50
CURR_EPOCH = 0

SAVE_FIGURES = True

MODEL_NAME = "UNet-MobileNetV2-FL-FaceMask"

np.random.seed(0)

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"./detectors/figures/{MODEL_NAME}/epoch{CURR_EPOCH}.jpg")
    # plt.show()

def create_mask(pred_mask):
    # pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask = pred_mask[..., tf.newaxis]
    # return pred_mask[0]
    pred_mask = pred_mask[0]
    pred_mask = tf.where(pred_mask>0.5,1,0)
    return pred_mask

# def show_predictions(dataset=None, num=1, sample_image=None, sample_mask=None):
#     if dataset:
#         for image, mask in dataset.take(num):
#             pred_mask = model.predict(image)
#             display([image[0], mask[0], create_mask(pred_mask)])
#     else:
#         display([sample_image, sample_mask,
#                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])
def show_predictions(dataset=None, num=1, sample_image=None, sample_mask=None):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image[:,:,:3], sample_mask,
                create_mask(model.predict(sample_image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, sampleI, sampleM):
        self.sample_image = sampleI
        self.sample_mask = sampleM

    def on_epoch_end(self, epoch, logs=None):
        global CURR_EPOCH
        CURR_EPOCH += 1
        if SAVE_FIGURES and CURR_EPOCH % 5 == 0:
            show_predictions(sample_image=self.sample_image, sample_mask=self.sample_mask)


if __name__ == "__main__":
    
    if not os.path.exists(f"./detectors/checkpoints/{MODEL_NAME}"):
        os.makedirs(f"./detectors/checkpoints/{MODEL_NAME}")
    if not os.path.exists(f"./detectors/figures/{MODEL_NAME}"):
        os.makedirs(f"./detectors/figures/{MODEL_NAME}")

    train_dataset = tf.data.Dataset.list_files(str(TRAIN_DATA_FOLDER/"*.png"))

    train_images = train_dataset.map(lambda x: load_image(x, True, IMAGE_HEIGHT, IMAGE_WIDTH), num_parallel_calls=tf.data.AUTOTUNE)

    trainData_size = len(train_dataset)

    train_indices = np.random.choice(range(trainData_size), int(VAL_RATIO * trainData_size), replace=False)
    print(f"Train size: {train_indices.shape[0]}")

    val_indices = list(set(range(trainData_size)) - set(train_indices))
    print(f"Validation size: {len(val_indices)}")

    train_images = subset_dataset(train_images, train_indices)
    val_images = subset_dataset(train_images, val_indices)

    TRAIN_LENGTH = len(train_indices)
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

    # for images, masks, faceMask in train_batches.take(1):
    #     sample_image, sample_mask, sample_faceMask = images[0], masks[0], faceMask[0]
        # display([sample_image, sample_mask, sample_faceMask])
    
    # val_batches = val_images.batch(BATCH_SIZE)
    val_batches = val_images.batch(BATCH_SIZE).map(AugmentAdditionalMask())

    # model = DeepLabV3(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS).get_model()
    model = UNet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS).get_model()

    # penultimate_layer = model.layers[-1]  # layer that you want to connect your new FC layer to 
    # new_top_layer = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(penultimate_layer.output)  # create new FC layer and connect it to the rest of the model
    # model = tf.keras.models.Model(model.input, new_top_layer)  # define your new model

    model.compile(optimizer='adam',
                # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False),
                # loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
                # loss = dice_loss,
                # loss_weights=[1,200],
                metrics=['accuracy'])

    # tf.keras.utils.plot_model(model, show_shapes=True)
    model.summary()

    # for images, masks in val_batches.take(1):
    #     sample_image, sample_mask = images[0], masks[0]

    for imageWithMask, masks in val_batches.take(1):
        sample_image = imageWithMask[0,:,:,:3]
        sample_mask = masks[0]

    checkPoint_callback = keras.callbacks.ModelCheckpoint("./detectors/checkpoints/"+MODEL_NAME+"/weights{epoch:04d}.h5",
                                        save_weights_only=False, period=10)

    # model_history = model.fit(train_batches, epochs=EPOCHS,
    #                         steps_per_epoch=STEPS_PER_EPOCH,
    #                         validation_data=val_batches,
    #                         callbacks=[DisplayCallback(sample_image, sample_mask), checkPoint_callback])
    model_history = model.fit(train_batches, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=val_batches,
                            callbacks=[DisplayCallback(imageWithMask[0], sample_mask), checkPoint_callback])

    history_dict = model_history.history
    json.dump(history_dict, open(f"./detectors/checkpoints/{MODEL_NAME}/modelHistory.json", 'w'))

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
    plt.savefig(f"./detectors/figures/{MODEL_NAME}/loss.jpg")