from tf_keras.models import Model
from tf_keras.layers import Input, Flatten, Dense, Dropout, Subtract, Concatenate, Activation, Layer, Conv2D, MaxPooling2D, BatchNormalization, Subtract, Activation
from tf_keras.regularizers import l1_l2, l2
import tensorflow as tf
import tensorflow.keras.backend as K
tf.random.set_seed(1234)

class snn:
    def __init__(self, numOfClass):
        self.n_class = numOfClass
        
    def __build_siamese_model(self, inputShape, res=True):
        inputs = Input(inputShape)

        #x = stn(inputs)
        #x = BatchNormalization()(inputs)
        x = Conv2D(32, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        feat1 = MaxPooling2D(pool_size=(2, 2))(x)
        feat1 = Dropout(0.5)(feat1)

        x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(feat1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        feat2 = MaxPooling2D(pool_size=(2, 2))(x)
        feat2 = Dropout(0.5)(feat2)


        x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(feat2)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        feat3 = MaxPooling2D(pool_size=(2, 2))(x)
        feat3 = Dropout(0.5)(feat3)


        feat1 = Flatten()(feat1)
        feat2 = Flatten()(feat2)
        feat3 = Flatten()(feat3)

        outputs = Concatenate()([feat1, feat2, feat3])
        model = Model(inputs, outputs)

        return model

    def get_model(self, input_shape, residual = True):
        imgA = Input(shape=input_shape)
        imgB = Input(shape=input_shape)
        featureExtractor1 = self.__build_siamese_model(input_shape,res=residual)

        featsA = featureExtractor1(imgA)
        featsB = featureExtractor1(imgB)
        distance = Subtract()([featsA, featsB])
        distance = AbsoluteLayer()(distance)


        if(self.n_class == 2):
            actv = "sigmoid"
            self.n_class = 1
        else:
            actv = "softmax"
            
        outputs = Dense(self.n_class, activation=actv, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(distance)
        model = Model(inputs=[imgA, imgB], outputs=outputs)

        return model

class AbsoluteLayer(Layer):
    def call(self, x):
        return tf.math.abs(x)

class SquareLayer(Layer):
    def call(self, x):
        return tf.math.square(x)

    # Example usage
input_shape = (64, 64, 3)  # Example input shape for image data
num_classes = 2
snn_model = snn(num_classes)
model = snn_model.get_model(input_shape)
model.summary()