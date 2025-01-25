from tf_keras.applications import ResNet50
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
        # Load ResNet50 model with pretrained weights, excluding the top layers
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=inputShape)

        # Freeze layers in ResNet50 if desired (set res=False if you want to fine-tune)
        # if res:
        #     for layer in base_model.layers:
        #         layer.trainable = False

        x = base_model.output
        x = Flatten()(x)  # Flatten the output for fully connected layers

        # Additional custom dense layers for feature extraction
        x = Dense(256, kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        feat4 = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        feat4 = BatchNormalization()(feat4)
        feat4 = Activation('relu')(feat4)

        model = Model(inputs=base_model.input, outputs=feat4)

        return model

    def get_model(self, input_shape, residual=True):
        imgA = Input(shape=input_shape)
        imgB = Input(shape=input_shape)
        featureExtractor = self.__build_siamese_model(input_shape, res=residual)

        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)

        # Compute absolute difference between feature vectors
        distance = Subtract()([featsA, featsB])
        distance = AbsoluteLayer()(distance)

        # Output layer configuration based on the number of classes
        if self.n_class == 2:
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

# Example usage
if __name__ == "__main__":
    input_shape = (16, 16, 1)  # Adjusted input shape for ResNet50
    num_classes = 2
    snn_model = snn(num_classes)
    model = snn_model.get_model(input_shape)
    model.summary()