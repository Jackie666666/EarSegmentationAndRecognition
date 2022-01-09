import tensorflow as tf

class VGG16():
    def __init__(self, imageH, imageW, imageC):
        self.image_height = imageH
        self.image_width = imageW
        self.image_channels = imageC

    def build_model(self):
        input_shape = (self.image_height, self.image_width, self.image_channels)
        base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
        base_model.trainable=False
        flatten = tf.keras.layers.Flatten()
        dropout1 = tf.keras.layers.Dropout(0.3)
        dense1 = tf.keras.layers.Dense(300, activation="relu")
        dropout2 = tf.keras.layers.Dropout(0.3)
        dense2 = tf.keras.layers.Dense(200, activation="relu")
        dropout3 = tf.keras.layers.Dropout(0.3)
        predictions = tf.keras.layers.Dense(100, activation="softmax")

        return tf.keras.Sequential([
            base_model,
            flatten,
            dropout1,
            dense1,
            dropout2,
            dense2,
            dropout3,
            predictions
        ])

class ResNet50():
    def __init__(self, imageH, imageW, imageC):
        self.image_height = imageH
        self.image_width = imageW
        self.image_channels = imageC

    def build_model(self):
        input_shape = (self.image_height, self.image_width, self.image_channels)
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
        base_model.trainable=False
        flatten = tf.keras.layers.Flatten()
        dropout1 = tf.keras.layers.Dropout(0.3)
        dense1 = tf.keras.layers.Dense(300, activation="relu")
        dropout2 = tf.keras.layers.Dropout(0.3)
        dense2 = tf.keras.layers.Dense(200, activation="relu")
        dropout3 = tf.keras.layers.Dropout(0.3)
        predictions = tf.keras.layers.Dense(100, activation="softmax")

        return tf.keras.Sequential([
            base_model,
            flatten,
            dropout1,
            dense1,
            dropout2,
            dense2,
            dropout3,
            predictions
        ])

class ResNet101():
    def __init__(self, imageH, imageW, imageC):
        self.image_height = imageH
        self.image_width = imageW
        self.image_channels = imageC

    def build_model(self):
        input_shape = (self.image_height, self.image_width, self.image_channels)
        base_model = tf.keras.applications.resnet_v2.ResNet101V2(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
        base_model.trainable=False
        flatten = tf.keras.layers.Flatten()
        dropout1 = tf.keras.layers.Dropout(0.3)
        dense1 = tf.keras.layers.Dense(300, activation="relu")
        dropout2 = tf.keras.layers.Dropout(0.3)
        dense2 = tf.keras.layers.Dense(200, activation="relu")
        dropout3 = tf.keras.layers.Dropout(0.3)
        predictions = tf.keras.layers.Dense(100, activation="softmax")

        return tf.keras.Sequential([
            base_model,
            flatten,
            dropout1,
            dense1,
            dropout2,
            dense2,
            dropout3,
            predictions
        ])


class DenseNet121():
    def __init__(self, imageH, imageW, imageC):
        self.image_height = imageH
        self.image_width = imageW
        self.image_channels = imageC

    def build_model(self):
        input_shape = (self.image_height, self.image_width, self.image_channels)
        base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
        base_model.trainable=False
        flatten = tf.keras.layers.Flatten()
        dropout1 = tf.keras.layers.Dropout(0.3)
        dense1 = tf.keras.layers.Dense(300, activation="relu")
        dropout2 = tf.keras.layers.Dropout(0.3)
        dense2 = tf.keras.layers.Dense(200, activation="relu")
        dropout3 = tf.keras.layers.Dropout(0.3)
        predictions = tf.keras.layers.Dense(100, activation="softmax")

        return tf.keras.Sequential([
            base_model,
            flatten,
            dropout1,
            dense1,
            dropout2,
            dense2,
            dropout3,
            predictions
        ])


class EfficientNetB0():
    def __init__(self, imageH, imageW, imageC):
        self.image_height = imageH
        self.image_width = imageW
        self.image_channels = imageC

    def build_model(self):
        input_shape = (self.image_height, self.image_width, self.image_channels)
        base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
        base_model.trainable=False
        flatten = tf.keras.layers.Flatten()
        dropout1 = tf.keras.layers.Dropout(0.3)
        dense1 = tf.keras.layers.Dense(300, activation="relu")
        dropout2 = tf.keras.layers.Dropout(0.3)
        dense2 = tf.keras.layers.Dense(200, activation="relu")
        dropout3 = tf.keras.layers.Dropout(0.3)
        predictions = tf.keras.layers.Dense(100, activation="softmax")

        return tf.keras.Sequential([
            base_model,
            flatten,
            dropout1,
            dense1,
            dropout2,
            dense2,
            dropout3,
            predictions
        ])