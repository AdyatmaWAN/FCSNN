from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Subtract, Concatenate, Activation
from tensorflow.keras.regularizers import l1_l2, l2
import tensorflow as tf
from keras import backend as K
tf.random.set_seed(1234)
class snn_1:
    def __init__(self, numOfClass):
        self.n_class = numOfClass

    def __build_siamese_model(self, inputShape, res=True):
        if(res):

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
            #feat1 = Dropout(0.5)(feat1)



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
            #feat2 = Dropout(0.5)(feat2)



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
            #feat3 = Dropout(0.5)(feat3)


            feat1 = Flatten()(feat1)
            feat2 = Flatten()(feat2)
            feat3 = Flatten()(feat3)

            x = Dense(256, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(feat3)
            x = BatchNormalization()(x)
            feat4 = Activation('relu')(x)

            x = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            feat4 = Activation('relu')(x)

            #feat4 = Dropout(0.5)(feat4)

            outputs = Concatenate()([feat1, feat2, feat3, feat4])
            model = Model(inputs, outputs)


        else:
            inputs = Input(inputShape)

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

            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.5)(x)


            x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.5)(x)


            x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.5)(x)


            outputs = Flatten()(x)
            '''
            x = Dense(256, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
            x = BatchNormalization()(x)
            outputs = Activation('relu')(x)
            
            #outputs = Dropout(0.5)(x) 
            
            x = Dense(256, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Dense(256, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
            x = BatchNormalization()(x)
            outputs = Activation('relu')(x)
            
            outputs = Dropout(0.5)(outputs)  
            '''
            model = Model(inputs, outputs)
        return model

    def get_model(self, input_shape, residual = True):
        imgA = Input(shape=input_shape)
        imgB = Input(shape=input_shape)
        featureExtractor1 = self.__build_siamese_model(input_shape,res=residual)

        featsA = featureExtractor1(imgA)
        featsB = featureExtractor1(imgB)
        distance = Subtract()([featsA, featsB])
        distance = K.abs(distance)


        if(self.n_class == 2):
            actv = "sigmoid"
            self.n_class = 1
        else:
            actv = "softmax"

        outputs = Dense(self.n_class, activation=actv, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(distance)
        model = Model(inputs=[imgA, imgB], outputs=outputs)

        return model
