import tensorflow as tf

class CustomNet():
    def __init__(self, imageH, imageW, imageC):
        self.image_height = imageH
        self.image_width = imageW
        self.image_channels = imageC

    # Source: https://github.com/chittalpatel/Music-Genre-Classification-GTZAN/blob/master/Music%20Genre%20Classification/CNN_train(1).ipynb
    def conv_block(self, x, n_filters,filter_size=(3, 3), pool_size=(2, 2),stride=(1, 1)):
        x = tf.keras.layers.Conv2D(n_filters, filter_size, strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=stride)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        return x

    def build_model(self):
        input_shape = (self.image_height, self.image_width, self.image_channels)
        inpt = tf.keras.layers.Input(shape=input_shape)
        x = self.conv_block(inpt, 16,stride=(2,2))
        x = self.conv_block(x, 32,filter_size=(3,3),stride=(2,2))
        x = self.conv_block(x, 64, stride=(2,2))
        x = self.conv_block(x, 128,filter_size=(3,3),stride=(2,2))
        x = self.conv_block(x, 256,stride=(2,2))

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(200, activation='relu', 
                kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        predictions = tf.keras.layers.Dense(100, 
                            activation='softmax',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        
        model = tf.keras.Model(inputs=inpt, outputs=predictions)
        return model