import tensorflow as tf

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask

def read_image(image_path, h, w):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    return tf.image.resize(image, [h, w])

def read_mask(mask_path, h, w):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    return tf.image.resize(mask, [h, w])

def load_image(image_path, isTrain, imageH, imageW):
    input_image = read_image(image_path, imageH, imageW)

    mask_path = tf.strings.regex_replace(image_path, 'train', "/annotations/segmentation/train") if isTrain \
        else tf.strings.regex_replace(image_path, 'test', "/annotations/segmentation/test")
    input_mask = read_mask(mask_path, imageH, imageW)

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