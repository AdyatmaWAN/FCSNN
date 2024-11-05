import tensorflow as tf
from tf_keras.layers import Conv2D, LayerNormalization, Dense, DepthwiseConv2D, GlobalAveragePooling2D, Layer, Reshape
from tf_keras.layers import Add, Multiply, Input, UpSampling2D, Dropout, ZeroPadding2D, Flatten,GlobalAveragePooling1D
from tf_keras.layers import Subtract, Activation, Concatenate, BatchNormalization
from tf_keras.regularizers import l1_l2
from tf_keras.models import Model
from tf_keras.activations import gelu, tanh
from tf_keras import Sequential
from typing import Optional, Tuple, List
from random import randint
import numpy as np
tf.keras.utils.set_random_seed(1234)


class PatchEmbed(Layer):
    """Image patch embedding layer, also acts as the down-sampling layer.

    Args:
        image_size (Tuple[int]): Input image resolution.
        patch_size (Tuple[int]): Patch spatial resolution.
        embed_dim (int): Embedding dimension.
    """

    def __init__(
            self,
            image_size: Tuple[int] = (16, 16),
            patch_size: Tuple[int] = (4, 4),
            embed_dim: int = 24,
            **kwargs,
    ):
        super().__init__(**kwargs)
        patch_resolution = [
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
            ]
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution[0] * patch_resolution[1]
        self.proj = Conv2D(
            filters=embed_dim, kernel_size=patch_size, strides=patch_size
        )
        self.flatten = Reshape(target_shape=(-1, embed_dim))
        self.norm = LayerNormalization(epsilon=1e-7)

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, int, int, int]:
        """Patchifies the image and converts into tokens.

        Args:
            x: Tensor of shape (B, H, W, C)

        Returns:
            A tuple of the processed tensor, height of the projected
            feature map, width of the projected feature map, number
            of channels of the projected feature map.
        """
        # Project the inputs.
        x = self.proj(x)

        # Obtain the shape from the projected tensor.
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]

        # B, H, W, C -> B, H*W, C
        x = self.norm(self.flatten(x))

        tf.debugging.check_numerics(x, message="NaN in PatchEmbed output")

        return x, height, width, channels

def MLP(in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, mlp_drop_rate: float = 0.0):
    hidden_features = hidden_features or in_features
    out_features = out_features or in_features

    return Sequential([
        Dense(units=hidden_features, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)),
        Dense(units=out_features, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)),
        Dropout(rate=mlp_drop_rate),
    ])


class FocalModulationLayer(Layer):
    def __init__(self, dim: int, focal_window: int, focal_level: int, focal_factor: int = 2, proj_drop_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.proj_drop_rate = proj_drop_rate

        # Use small initial values for Dense layers
        self.initial_proj = Dense(
            units=(2 * self.dim) + (self.focal_level + 1),
            use_bias=True,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )

        self.focal_layers = []
        for idx in range(self.focal_level):
            kernel_size = (self.focal_factor * idx) + self.focal_window
            layer = Sequential([
                ZeroPadding2D(padding=(kernel_size // 2, kernel_size // 2)),
                Conv2D(
                    filters=self.dim,
                    kernel_size=kernel_size,
                    activation=gelu,
                    groups=self.dim,
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
                ),
            ])
            self.focal_layers.append(layer)

        # Smaller activation range using tanh instead of gelu
        self.activation = tanh
        self.gap = GlobalAveragePooling2D(keepdims=True)
        self.modulator_proj = Conv2D(
            filters=self.dim,
            kernel_size=(1, 1),
            use_bias=True,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )
        self.proj = Dense(units=self.dim, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.proj_drop = Dropout(self.proj_drop_rate)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x_proj = self.initial_proj(x)
        tf.debugging.check_numerics(x_proj, message="NaN after initial projection in FocalModulationLayer")

        query, context, self.gates = tf.split(value=x_proj, num_or_size_splits=[self.dim, self.dim, self.focal_level + 1], axis=-1)
        self.gates = tf.clip_by_value(self.gates, -1.0, 1.0)

        context = self.focal_layers[0](context)
        context_all = context * self.gates[..., 0:1]

        for idx in range(1, self.focal_level):
            context = self.focal_layers[idx](context)
            context_all += context * self.gates[..., idx: idx + 1]

        context_global = self.activation(self.gap(context))  # Replace with bounded activation
        context_all += context_global * self.gates[..., self.focal_level :]

        self.modulator = self.modulator_proj(context_all)

        # Clipping to avoid extreme values before multiplication
        query = tf.clip_by_value(query, -1e2, 1e2)
        self.modulator = tf.clip_by_value(self.modulator, -1e2, 1e2)

        x_output = query * self.modulator
        x_output = self.proj(x_output)
        x_output = self.proj_drop(x_output)

        return x_output


class FocalModulationBlock(Layer):
    """Combine FFN and Focal Modulation Layer.

    Args:
        dim (int): Number of input channels.
        input_resolution (Tuple[int]): Input resulotion.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float): Dropout rate.
        drop_path (float): Stochastic depth rate.
        focal_level (int): Number of focal levels.
        focal_window (int): Focal window size at first focal level
    """

    def __init__(
            self,
            dim: int,
            input_resolution: Tuple[int],
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            drop_path: float = 0.0,
            focal_level: int = 1,
            focal_window: int = 3,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.norm = LayerNormalization(epsilon=1e-5)
        self.modulation = FocalModulationLayer(
            dim=self.dim,
            focal_window=self.focal_window,
            focal_level=self.focal_level,
            proj_drop_rate=drop,
        )
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            mlp_drop_rate=drop,
        )

    def call(self, x: tf.Tensor, height: int, width: int, channels: int) -> tf.Tensor:
        """Processes the input tensor through the focal modulation block.

        Args:
            x (tf.Tensor): Inputs of the shape (B, L, C)
            height (int): The height of the feature map
            width (int): The width of the feature map
            channels (int): The number of channels of the feature map

        Returns:
            The processed tensor.
        """
        shortcut = x

        # Focal Modulation
        x = tf.reshape(x, shape=(-1, height, width, channels))
        x = self.modulation(x)
        x = tf.reshape(x, shape=(-1, height * width, channels))
        tf.debugging.check_numerics(x, message="NaN after focal modulation in FocalModulationBlock")

        # FFN
        x = shortcut + x
        # tf.print(x)
        tf.debugging.check_numerics(x, message="NaN before FFN in FocalModulationBlock")
        x = x + self.mlp(self.norm(x))
        # tf.print(x)
        tf.debugging.check_numerics(x, message="NaN after FFN in FocalModulationBlock")
        return x

class BasicLayer(Layer):
    """Collection of Focal Modulation Blocks."""

    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: Tuple[int],
            depth: int,
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            downsample=None,
            focal_level: int = 1,
            focal_window: int = 1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = [
            FocalModulationBlock(
                dim=dim,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                focal_level=focal_level,
                focal_window=focal_window,
            )
            for i in range(self.depth)
        ]

        # Downsample layer at the end of the layer
        if downsample is not None:
            self.downsample = downsample(
                image_size=input_resolution,
                patch_size=(2, 2),
                embed_dim=out_dim,
            )
        else:
            self.downsample = None

    def call(
            self, x: tf.Tensor, height: int, width: int, channels: int
    ) -> Tuple[tf.Tensor, tf.Tensor, int, int, int]:
        """Forward pass of the layer, returning pre-downsample residual as well.

        Args:
            x (tf.Tensor): Tensor of shape (B, L, C)
            height (int): Height of feature map
            width (int): Width of feature map
            channels (int): Embed Dim of feature map

        Returns:
            A tuple of (processed tensor before downsampling, downsampled tensor, height, width, dim).
        """
        # Apply Focal Modulation Blocks
        for block in self.blocks:
            x = block(x, height, width, channels)

        # Save the residual output before downsampling
        residual = x

        # Downsample, if applicable
        if self.downsample is not None:
            x = tf.reshape(x, shape=(-1, height, width, channels))
            x, height_o, width_o, channels_o = self.downsample(x)
        else:
            height_o, width_o, channels_o = height, width, channels

        return residual, x, height_o, width_o, channels_o


class FocalModulationNetwork(Model):
    """The Focal Modulation Network with concatenated pre-downsample residuals."""

    def __init__(
            self,
            image_size: Tuple[int] = (16, 16),
            patch_size: Tuple[int] = (4, 4),
            embed_dim: int = 256,
            depths: List[int] = [2, 2, 6],
            mlp_ratio: float = 4.0,
            drop_rate: float = 0.1,
            focal_levels=[2, 2, 2],
            focal_windows=[3, 3, 3],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2**i) for i in range(self.num_layers)]
        self.embed_dim = embed_dim
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim[0],
        )
        self.pos_drop = Dropout(drop_rate)
        self.basic_layers = []

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim[i_layer],
                out_dim=embed_dim[i_layer + 1]
                if (i_layer < self.num_layers - 1)
                else None,
                input_resolution=(
                    self.patch_embed.patch_resolution[0] // (2**i_layer),
                    self.patch_embed.patch_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
            )
            self.basic_layers.append(layer)

        self.norm = LayerNormalization(epsilon=1e-7)
        self.avgpool = GlobalAveragePooling1D()
        self.flatten = Flatten()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass of the network with concatenated pre-downsample residuals.

        Args:
            x: Tensor of shape (B, H, W, C)

        Returns:
            Concatenated residuals as a single vector.
        """
        # Patch Embed the input images.
        x, height, width, channels = self.patch_embed(x)
        x = self.pos_drop(x)

        # List to store flattened residuals from each BasicLayer
        residuals = []

        for idx, layer in enumerate(self.basic_layers):
            # print(idx)
            # Get pre-downsampled (residual) and downsampled outputs
            residual, x, height, width, channels = layer(x, height, width, channels)

            # Flatten the residual and add it to the list
            # print("residual", residual.shape)
            # print("x", x.shape)
            residual_flattened = Flatten()(residual)
            # print("residual_flattened", residual_flattened.shape)
            residuals.append(residual_flattened)
            # print("residuals", len(residuals))
            # print()
            tf.debugging.check_numerics(residual_flattened, message="NaN in residual_flattened in FocalModulationNetwork")

        # Concatenate all flattened residuals
        concatenated_residuals = tf.concat(residuals, axis=-1)
        tf.debugging.check_numerics(concatenated_residuals, message="NaN in concatenated residuals in FocalModulationNetwork")
        # Optional: apply normalization and pooling if needed
        concatenated_residuals = self.norm(concatenated_residuals)
        # tf.print(concatenated_residuals)
        # tf.print(concatenated_residuals.shape)
        # tf.print()
        return concatenated_residuals


class AbsoluteLayer(Layer):
    def call(self, x):
        return tf.math.abs(x)

class SquareLayer(Layer):
    def call(self, x):
        return tf.math.square(x)

class snn:
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.feature_extractor_1 = self.build_feature_extractor()
        self.feature_extractor_2 = self.build_feature_extractor()

    def build_feature_extractor(self):
        """Creates the base feature extraction model using FocalModulationNetwork."""
        return FocalModulationNetwork(
            image_size=(16, 16),  # Example image size, adjust as needed
            embed_dim=256,  # Adjusted embed_dim for simplicity
            depths=[2, 2, 6],
            mlp_ratio=4.0,
            drop_rate=0.1,
            focal_levels=[2, 2, 2],
            focal_windows=[3, 3, 3]
        )

    def get_model(self, input_shape: Tuple[int, int, int], residual=True):
        """Creates the Siamese network with an MLP classifier."""

        # Define the two input images for the Siamese network
        imgA = Input(shape=input_shape)
        imgB = Input(shape=input_shape)

        # Use the feature extractor for both inputs
        featsA = self.feature_extractor_1(imgA)
        featsA = AbsoluteLayer()(featsA)  # Apply absolute value

        featsB = self.feature_extractor_2(imgB)
        featsB = AbsoluteLayer()(featsB)  # Apply absolute value

        tf.debugging.check_numerics(featsA, message="NaN found in featsA")
        tf.debugging.check_numerics(featsB, message="NaN found in featsB")

        # Compute the absolute difference between features
        distance = Subtract()([featsA, featsB])
        distance = AbsoluteLayer()(distance)  # Apply absolute value
        distance = SquareLayer()(distance)  # Apply square
        tf.debugging.check_numerics(distance, message="NaN found in distance after AbsoluteLayer")
        # Add an MLP for classification
        if self.num_classes == 2:
            activation = "sigmoid"
            output_units = 1  # For binary classification
        else:
            activation = "softmax"
            output_units = self.num_classes

        # Output dense layer with regularization
        outputs = Dense(
            units=output_units,
            activation=activation,
            # kernel_regularizer=l1_l2(0.01),
            # bias_regularizer=l1_l2(0.01)
        )(distance)
        tf.debugging.check_numerics(outputs, message="NaN found in final outputs")
        # Define the full model
        model = Model(inputs=[imgA, imgB], outputs=outputs)
        return model

# Example usage
if __name__ == "__main__":
    input_shape = (16, 16, 1)  # Adjust input shape as needed for your data
    num_classes = 2  # Adjust based on binary or multi-class classification
    siamese_network = snn(num_classes=num_classes)
    model = siamese_network.get_model(input_shape)
    model.summary()