import tensorflow as tf
import detectors.my_detectors.pix2pix as pix2pix
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Add
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense, LeakyReLU, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from EfficientUnet.efficientunet.efficientunet import get_efficient_unet_b0

class UNet():
    def __init__(self, imageH, imageW, imageC):
        self.image_height = imageH
        self.image_width = imageW
        self.image_channels = imageC

    def get_model(self):
        model = get_efficient_unet_b0((self.image_height, self.image_width, self.image_channels),
            pretrained=True, block_type='transpose', concat_input=True, out_channels=2)
        return model
 


    # Unet-MobileNet-FocalLoss
    #     self.encoder = self.get_encoder()
    
    # def get_encoder(self):
    #     # to pass more channels
    #     base_model = tf.keras.applications.MobileNetV2(input_shape=(self.image_height, self.image_width, self.image_channels),
    #                                                   include_top=False,
    #                                                   weights=None)
    #     base_weights = tf.keras.applications.MobileNetV2(input_shape=(self.image_height, self.image_width, 3),
    #                                                      include_top=False,
    #                                                      weights='imagenet')

    #     for i in range(2, len(base_model.layers)):
    #         base_model.layers[i].set_weights(base_weights.layers[i].get_weights())

    #     del base_weights

    #     # base_model = tf.keras.applications.MobileNetV2(input_shape=[self.image_height, self.image_width, 3], include_top=False)
    #     # Use the activations of these layers
    #     layer_names = [
    #         'block_1_expand_relu',   # 64x64
    #         'block_3_expand_relu',   # 32x32
    #         'block_6_expand_relu',   # 16x16
    #         'block_13_expand_relu',  # 8x8
    #         'block_16_project',      # 4x4
    #     ]
    #     base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    #     # Create the feature extraction model
    #     down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    #     down_stack.trainable = False

    #     return down_stack


    # def get_model(self):
    #     up_stack = [
    #         pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    #         pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    #         pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    #         pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    #     ]
        
    #     inputs = tf.keras.layers.Input(shape=[self.image_height, self.image_width, self.image_channels])
    #     # Downsampling through the model
    #     skips = self.encoder(inputs)
    #     x = skips[-1]
    #     skips = reversed(skips[:-1])

    #     # Upsampling and establishing the skip connections
    #     for up, skip in zip(up_stack, skips):
    #         x = up(x)
    #         concat = tf.keras.layers.Concatenate()
    #         x = concat([x, skip])

    #     # This is the last layer of the model
    #     secondLast = tf.keras.layers.Conv2DTranspose(
    #         filters=2, kernel_size=3, strides=2,
    #         padding='same')  #64x64 -> 128x128
        
    #     x = secondLast(x)
    #     last = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")
    #     x = last(x)

    #     return tf.keras.Model(inputs=inputs, outputs=x)


