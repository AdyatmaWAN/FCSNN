from tf_keras.models import Model
from tf_keras.layers import Input, Flatten, Dense, Dropout, Subtract, Concatenate, Activation, Layer, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Multiply, Add
from tf_keras.regularizers import l1_l2, l2
import tensorflow as tf

tf.random.set_seed(1234)

class SEBlock(Layer):
    def __init__(self, filters, reduction=16):
        super(SEBlock, self).__init__()
        self.filters = filters
        self.reduction = reduction

    def build(self, input_shape):
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(self.filters // self.reduction, activation='relu', kernel_initializer='he_normal')
        self.dense2 = Dense(self.filters, activation='sigmoid', kernel_initializer='he_normal')

    def call(self, x):
        se = self.global_avg_pool(x)
        se = Reshape((1, 1, self.filters))(se)
        se = self.dense1(se)
        se = self.dense2(se)
        return Multiply()([x, se])

class CBAMBlock(Layer):
    def __init__(self, filters, reduction=16):
        super(CBAMBlock, self).__init__()
        self.filters = filters
        self.reduction = reduction

    def build(self, input_shape):
        # Channel Attention
        # self.channel_attention = SEBlock(self.filters, self.reduction)

        # Spatial Attention
        self.global_avg_pool = GlobalAveragePooling2D()
        self.global_max_pool = GlobalMaxPooling2D()
        self.dense1 = Dense(self.filters // self.reduction, activation='relu')
        self.dense2 = Dense(self.filters, activation='sigmoid')

        # Define the Conv2D layer only once
        self.spatial_conv = Conv2D(1, (7, 7), padding="same", activation='sigmoid')

    def spatial_attention(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])

        # Use the pre-defined Conv2D layer here
        return self.spatial_conv(concat)

    def call(self, x):
        # Channel Attention
        # x = self.channel_attention(x)

        # Spatial Attention
        sa = self.spatial_attention(x)
        return Multiply()([x, sa])


class snn:
    def __init__(self, numOfClass):
        self.n_class = numOfClass

    def __build_siamese_model(self, inputShape, res=True):
        inputs = Input(inputShape)

        x = Conv2D(32, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = SEBlock(32)(x)  # Add SE Block
        x = CBAMBlock(32)(x)  # Add CBAM Block

        x = Conv2D(32, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = SEBlock(32)(x)
        x = CBAMBlock(32)(x)

        x = Conv2D(32, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = SEBlock(32)(x)
        x = CBAMBlock(32)(x)

        feat1 = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(feat1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = SEBlock(64)(x)
        x = CBAMBlock(64)(x)

        x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = SEBlock(64)(x)
        x = CBAMBlock(64)(x)

        x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = SEBlock(64)(x)
        x = CBAMBlock(64)(x)

        feat2 = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(feat2)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = SEBlock(128)(x)
        x = CBAMBlock(128)(x)

        x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = SEBlock(128)(x)
        x = CBAMBlock(128)(x)

        x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = SEBlock(128)(x)
        x = CBAMBlock(128)(x)

        feat3 = MaxPooling2D(pool_size=(2, 2))(x)

        feat1 = Flatten()(feat1)
        feat2 = Flatten()(feat2)
        feat3 = Flatten()(feat3)

        x = Dense(256, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(feat3)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        feat4 = Activation('relu')(x)

        outputs = Concatenate()([feat1, feat2, feat3, feat4])
        model = Model(inputs, outputs)

        return model

    def get_model(self, input_shape, residual=True):
        imgA = Input(shape=input_shape)
        imgB = Input(shape=input_shape)
        featureExtractor = self.__build_siamese_model(input_shape, res=residual)

        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)
        distance = Subtract()([featsA, featsB])
        distance = AbsoluteLayer()(distance)

        actv = "sigmoid" if self.n_class == 2 else "softmax"
        output_units = 1 if self.n_class == 2 else self.n_class

        outputs = Dense(output_units, activation=actv, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(distance)
        model = Model(inputs=[imgA, imgB], outputs=outputs)

        return model

class AbsoluteLayer(Layer):
    def call(self, x):
        return tf.math.abs(x)

class SquareLayer(Layer):
    def call(self, x):
        return tf.math.square(x)

# Example usage
if __name__ == "__main__":
    input_shape = (16, 16, 1)  # Updated input shape for a single-channel (1-channel) image
    num_classes = 2
    snn_model = snn(num_classes)
    model = snn_model.get_model(input_shape)
    model.summary()