import tensorflow as tf
import detectors.my_detectors.pix2pix as pix2pix
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50

class UNet():
    def __init__(self, imageH, imageW, imageC):
        self.image_height = imageH
        self.image_width = imageW
        self.image_channels = imageC

        # Unet-EfficientNetb0
    def conv_block(self, inputs, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def decoder_block(self, inputs, skip, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = Concatenate()([x, skip])
        x = self.conv_block(x, num_filters)
        return x

    def get_model(self):
        #  Input 
        inputs = Input((self.image_height, self.image_width, self.image_channels))

        #  Pre-trained Encoder 
        encoder = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
        encoder.trainable = False
        s1 = encoder.get_layer("input_1").output                      ## 256
        s2 = encoder.get_layer("block2a_expand_activation").output    ## 128
        s3 = encoder.get_layer("block3a_expand_activation").output    ## 64
        s4 = encoder.get_layer("block4a_expand_activation").output    ## 32

        #  Bottleneck 
        b1 = encoder.get_layer("block6a_expand_activation").output    ## 16

        #  Decoder 
        d1 = self.decoder_block(b1, s4, 512)                               ## 32
        d2 = self.decoder_block(d1, s3, 256)                               ## 64
        d3 = self.decoder_block(d2, s2, 128)                               ## 128
        d4 = self.decoder_block(d3, s1, 64)                                ## 256

        #  Output 
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

        model = Model(inputs, outputs, name="EfficientNetB0_UNET")
        return model
"""
    # Unet-MobileNet-FocalLoss
        self.encoder = self.get_encoder()
    
    def get_encoder(self):
        base_model = tf.keras.applications.MobileNetV2(input_shape=[self.image_height, self.image_width, self.image_channels], include_top=False)
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
  
        return down_stack


    def get_model(self):
        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]
        
        inputs = tf.keras.layers.Input(shape=[self.image_height, self.image_width, self.image_channels])
        # Downsampling through the model
        skips = self.encoder(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        secondLast = tf.keras.layers.Conv2DTranspose(
            filters=2, kernel_size=3, strides=2,
            padding='same')  #64x64 -> 128x128
        
        x = secondLast(x)
        last = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")
        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
"""

