import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from tensorflow import keras
import tensorflow_addons as tfa

BASE_PATH = "../../data"

TRAIN_DATA_FOLDER = pathlib.Path(BASE_PATH + "/train")
TEST_DATA_FOLDER = pathlib.Path(BASE_PATH + "/test")

TRAIN_MASK_FOLDER = pathlib.Path(BASE_PATH + "/annotations/segmentation/train")
TEST_MASK_FOLDER = pathlib.Path(BASE_PATH + "/annotations/segmentation/train")

# IMAGE_HEIGHT = 352 
# IMAGE_WIDTH = 480
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

CLASSES = 2
VAL_RATIO = 0.85
BATCH_SIZE = 8
EPOCHS = 30
CURR_EPOCH = 0

MODEL_NAME = "UNet-MobileNet"

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask

def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    return tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])

def read_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    return tf.image.resize(mask, [IMAGE_HEIGHT, IMAGE_WIDTH])

def load_image(image_path, isTrain):
    input_image = read_image(image_path)

    mask_path = tf.strings.regex_replace(image_path, 'train', "/annotations/segmentation/train") if isTrain \
        else tf.strings.regex_replace(image_path, 'test', "/annotations/segmentation/test")
    input_mask = read_mask(mask_path)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def subset_dataset(dataset, indices):
    return dataset.enumerate().filter(lambda i, t: tf.reduce_any(i == indices)).map(lambda j, u: u)

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.HSV = [0.3, 0.3, 0.3]

    def hsv(self, image, mask):
        if tf.random.uniform([]) < self.HSV[0]:
            image = tf.image.adjust_hue(image, tf.random.uniform([], -0.1, 0.1))
        if tf.random.uniform([]) < self.HSV[1]:
            image = tf.image.adjust_saturation(image, tf.random.uniform([], 0, 2))
        if tf.random.uniform([]) < self.HSV[2]:
            image = tf.image.adjust_brightness(image, tf.random.uniform([], -0.2, 0.2))
        return image, mask

    def call(self, inputs, labels):
        # inputs = tfa.image.equalize(inputs)
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        inputs, labels = self.hsv(inputs, labels)
        return inputs, labels


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"../figures/{MODEL_NAME}/epoch{CURR_EPOCH}.jpg")

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1, sample_image=None, sample_mask=None):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                create_mask(model.predict(sample_image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, sampleI, sampleM):
        self.sample_image = sampleI
        self.sample_mask = sampleM

    def on_epoch_end(self, epoch, logs=None):
        global CURR_EPOCH
        CURR_EPOCH += 1
        show_predictions(sample_image=self.sample_image, sample_mask=self.sample_mask)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    
    if not os.path.exists(f"../checkpoints/{MODEL_NAME}"):
        os.makedirs(f"../checkpoints/{MODEL_NAME}")
    if not os.path.exists(f"../figures/{MODEL_NAME}"):
        os.makedirs(f"../figures/{MODEL_NAME}")

    train_dataset = tf.data.Dataset.list_files(str(TRAIN_DATA_FOLDER/"*.png"))
    test_dataset = tf.data.Dataset.list_files(str(TEST_DATA_FOLDER/"*.png"))

    train_images = train_dataset.map(lambda x: load_image(x, True), num_parallel_calls=tf.data.AUTOTUNE)
    test_images = test_dataset.map(lambda x: load_image(x, False), num_parallel_calls=tf.data.AUTOTUNE)

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


    val_batches = val_images.batch(BATCH_SIZE)
    test_batches = test_images.batch(BATCH_SIZE)

    for images, masks in train_batches.take(2):
        sample_image, sample_mask = images[0], masks[0]
    #     display([sample_image, sample_mask])


    base_model = tf.keras.applications.MobileNetV2(input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    import pix2pix

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    model = unet_model(output_channels=CLASSES)
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                loss_weights=[1,200],
                metrics=['accuracy'])

    # tf.keras.utils.plot_model(model, show_shapes=True)
    model.summary()

    # for images, masks in train_batches.take(1):
    #     sample_image, sample_mask = images[0], masks[0]
    #     show_predictions(sample_image=sample_image, sample_mask=sample_mask)

    chechPoint_callback = keras.callbacks.ModelCheckpoint("../checkpoints/"+MODEL_NAME+"/weights{epoch:04d}.h5",
                                        save_weights_only=False, period=5)

    model_history = model.fit(train_batches, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=val_batches,
                            callbacks=[DisplayCallback(sample_image, sample_mask), chechPoint_callback])

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../figures/{MODEL_NAME}/loss.jpg")