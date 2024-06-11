# +
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten, Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Subtract, Concatenate, Activation
from tensorflow.keras.regularizers import l1_l2, l2
import tensorflow as tf
from keras import backend as K
tf.random.set_seed(1234)

class squared_substraction(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(quared_substraction, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, inputs):
        a = inputs[0]
        b = inputs[1]
        
        return 0

class snn:
    def __init__(self, numOfClass):
        self.n_class = numOfClass

    def __get_block(self):
        pass
    def __ResidualBlock(self, width):
        penalty = 0.1
        def apply(x):
            input_width = x.shape[3]
            if input_width == width:
                residual = x
            else:
                residual = layers.Conv2D(width, kernel_size=1, kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(x)
            x = BatchNormalization(center=False, scale=False)(x)
            x = Conv2D(width, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(x)
            x = Conv2D(width, kernel_size=3, padding="same", kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(x)
            x = Add()([x, residual])
            return x
        return apply

    def __build_siamese_model(self, inputShape, res=True):
        if(res):
        
            inputs = Input(inputShape)
            
            penalty = 0.01

            x = Conv2D(32, (3, 3), strides=1, padding="same", kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(inputs)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
            x = self.__ResidualBlock(32)(x)
            x = Activation('relu')(x)
            feat1 = MaxPooling2D(pool_size=(2, 2))(x)
            #feat1 = Dropout(0.5)(feat1)
            
            x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(feat1)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x) 
            x = self.__ResidualBlock(64)(x)
            x = Activation('relu')(x)
            feat2 = MaxPooling2D(pool_size=(2, 2))(x)
            #feat2 = Dropout(0.5)(feat2)
              
            x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(feat2)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
            x = self.__ResidualBlock(128)(x)
            x = Activation('relu')(x)
            feat3 = MaxPooling2D(pool_size=(2, 2))(x)
            #feat3 = Dropout(0.5)(feat3)     

            
            x = Conv2D(256, (3, 3), strides=1, padding="same", kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(feat3)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
            x = self.__ResidualBlock(256)(x)
            x = Activation('relu')(x)
            feat4 = MaxPooling2D(pool_size=(2, 2))(x)
            #feat4 = Dropout(0.5)(feat4)  
            
            x = Conv2D(512, (3, 3), strides=1, padding="same", kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(feat4)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
            x = self.__ResidualBlock(512)(x)
            x = Activation('relu')(x)
            feat5 = MaxPooling2D(pool_size=(2, 2))(x)
            #feat5= Dropout(0.5)(feat5)  
            
            x = Conv2D(512, (3, 3), strides=1, padding="same", kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(feat5)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
            x = self.__ResidualBlock(512)(x)
            x = Activation('relu')(x)
            feat6 = MaxPooling2D(pool_size=(2, 2))(x)
            #feat6= Dropout(0.5)(feat6)  
            
            
            feat1 = Flatten()(feat1)
            feat2 = Flatten()(feat2)
            feat3 = Flatten()(feat3)
            feat4 = Flatten()(feat4)
            feat5 = Flatten()(feat5)
            feat6 = Flatten()(feat6)
            '''
            x = Dense(256, kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(feat3)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
    
            x = Dense(256, kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(x)
            x = BatchNormalization()(x)
            feat4 = Activation('relu')(x)
            '''
            
            #feat4 = Dropout(0.5)(feat4)  
            
            outputs = Concatenate()([feat1, feat2, feat3, feat4, feat5, feat6])
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
            
            x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=1, padding="same", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.5)(x) 
            

            x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=1, padding="same", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
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
        penalty = 0.01
        imgA = Input(shape=input_shape)
        imgB = Input(shape=input_shape)
        featureExtractor1 = self.__build_siamese_model(input_shape,res=residual)
        featureExtractor2 = self.__build_siamese_model(input_shape,res=residual)

        featsA = featureExtractor1(imgA)
        featsB = featureExtractor2(imgB)
        
        
        distance = (featsA - featsB) ** 2
# #         distance = Subtract()([featsA, featsB])
# #         distance = K.abs(distance)

#         distance = tf.math.reduce_mean((featsA - featsB) ** 2, keepdims=True, axis=1)


        if(self.n_class == 2):
            actv = "sigmoid"
            self.n_class = 1
        else:
            actv = "softmax"
            
        outputs = Dense(self.n_class, activation=actv, kernel_regularizer=l2(penalty), bias_regularizer=l2(penalty))(distance)
        model = Model(inputs=[imgA, imgB], outputs=outputs)

        return model
