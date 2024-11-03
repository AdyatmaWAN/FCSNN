from tf_keras.models import Model
from tf_keras.layers import Input, Dense, Dropout, Subtract, Concatenate, Activation, Layer
from tf_keras.regularizers import l1_l2
import tensorflow as tf
import tensorflow.keras.backend as K
from tf_keras.layers import LayerNormalization, MultiHeadAttention

class snn_vit:
    def __init__(self, numOfClass):
        self.n_class = numOfClass

    def __build_vit_model(self, inputShape, num_heads=8, embed_dim=64, depth=6, patch_size=4):
        inputs = Input(inputShape)
        # Extract patches from the input image
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # Flatten the patches
        patch_dims = patches.shape[-1]
        x = tf.reshape(patches, [-1, patches.shape[1] * patches.shape[2], patch_dims])

        # Linear Projection and Layer Normalization
        x = Dense(embed_dim)(x)
        x = LayerNormalization()(x)

        # Positional Embedding (Sine-Cosine)
        position_embedding = self.__posemb_sincos_2d(patches.shape[1], patches.shape[2], embed_dim)
        position_embedding = tf.cast(position_embedding, x.dtype)
        x += position_embedding

        # Transformer Encoder Blocks
        for _ in range(depth):
            # Multi-Head Self-Attention
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
            # Residual connection and normalization
            x = LayerNormalization()(x + attention_output)

            # Feed Forward Network
            ffn = Dense(embed_dim * 4, activation='relu')(x)
            ffn = Dense(embed_dim)(ffn)
            # Residual connection and normalization
            x = LayerNormalization()(x + ffn)

        # Final embedding representation
        x = tf.reduce_mean(x, axis=1)
        x = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01))(x)
        x = Dropout(0.5)(x)
        model = Model(inputs, x)
        return model

    def __posemb_sincos_2d(self, h, w, dim, temperature=10000):
        y, x = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")
        assert (dim % 4) == 0, "Feature dimension must be multiple of 4 for sincos embedding"
        omega = tf.range(dim // 4, dtype=tf.float32) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = tf.cast(tf.reshape(y, [-1, 1]), dtype=tf.float32) * omega
        x = tf.cast(tf.reshape(x, [-1, 1]), dtype=tf.float32) * omega
        pe = tf.concat([tf.sin(x), tf.cos(x), tf.sin(y), tf.cos(y)], axis=1)
        return tf.reshape(pe, [h * w, dim])

    def get_model(self, input_shape, residual=True):
        imgA = Input(shape=input_shape)
        imgB = Input(shape=input_shape)
        featureExtractor = self.__build_vit_model(input_shape)

        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)
        distance = Subtract()([featsA, featsB])
        distance = AbsoluteLayer()(distance)

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

class SquareLayer(Layer):
    def call(self, x):
        return tf.math.square(x)

# Example usage
input_shape = (64, 64, 3)  # Example input shape for image data
num_classes = 2
snn_model = snn_vit(num_classes)
model = snn_model.get_model(input_shape)
model.summary()
