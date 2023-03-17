import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential


class Block:
    """Base class for all blocks"""

    def __init__(self, name):
        self.name = name

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(name={self.name})"

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        raise NotImplementedError


class Residual(Block):
    """The simple Residual block"""

    def __init__(
        self,
        name,
        filters,
        kernel_size=3,
        expand="B",
        strides=1,
        activation="relu",
    ):
        super().__init__(name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.expand = expand
        self.strides = strides
        self.activation = activation

    def __str__(self) -> str:
        return "Residual Block"

    def __repr__(self) -> str:
        return (
            super().__repr__()[:-1]
            + f", filters={self.filters}, kernel_size={self.kernel_size}, strides={self.strides}, activation={self.activation})"
        )

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        """The call method for the block

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor

        Returns
        -------
        (tf.keras.Model, tf.Tensor)
            The block model and the output tensor
        """
        inputs_ = inputs
        x = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            name=self.name + "_conv1",
        )(inputs)
        x = layers.BatchNormalization(name=self.name + "_bn1")(x)
        x = layers.Activation(self.activation, name=self.name + "_act1")(x)

        x = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="same",
            name=self.name + "_conv2",
        )(x)
        x = layers.BatchNormalization(name=self.name + "_bn2")(x)

        if self.strides != 1:
            # If the stride is not 1, downsample the input to match the output shape
            if self.expand == "projection" or self.expand == "B":
                # Projection expansion
                inputs = layers.Conv2D(
                    filters=self.filters,
                    kernel_size=1,
                    strides=self.strides,
                    padding="same",
                    name=f"{self.name}_projection",
                )(inputs)
                inputs = layers.BatchNormalization(name=f"{self.name}_bn3")(inputs)

            elif self.expand == "zero" or self.expand == "A":
                # Zero padding expansion
                def pad_depth(x, desired_channels):
                    """Pads the depth of the input tensor to match the desired channels"""
                    y = tf.zeros_like(x, name=f"{self.name}_pad_depth1")
                    new_channels = desired_channels - x.shape.as_list()[-1]
                    y = y[..., :new_channels]
                    return layers.concatenate([x, y], name=f"{self.name}_pad_depth2")

                desired_channels = x.shape.as_list()[-1]
                inputs = layers.MaxPool2D(
                    pool_size=(1, 1),
                    strides=(2, 2),
                    padding="same",
                    name=f"{self.name}_zero_upscale_pool",
                )(inputs)
                inputs = layers.Lambda(
                    pad_depth,
                    arguments={"desired_channels": desired_channels},
                    name=f"{self.name}_zero_upscale_lambda",
                )(inputs)
                inputs = layers.BatchNormalization(name=f"{self.name}_bn3")(inputs)

        x = layers.Add(name=self.name + "_add")([x, inputs])
        x = layers.Activation(self.activation, name=self.name + "_act2")(x)
        block = Model(inputs=inputs_, outputs=x, name=self.name)
        return block, x

    def summary(self, input_shape):
        """Prints a summary of the block"""
        if input_shape[-1] != self.filters:
            print(
                "WARNING: The input shape does not match the number of filters in the block.\nSummary will work but the output shape will be wrong."
            )
            input_shape = (input_shape[0], input_shape[1], self.filters)
        inputs = layers.Input(shape=input_shape)
        block, _ = self(inputs)
        block.summary()


class ResidualBottleneck(Block):
    """The Residual bottleneck block"""

    def __init__(
        self,
        name,
        filters,
        kernel_size=3,
        strides=1,
        r=4,
        activation="relu",
    ):
        super().__init__(name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.r = r
        self.activation = activation

    def __str__(self) -> str:
        return "Residual Bottleneck Block"

    def __repr__(self) -> str:
        return (
            super().__repr__()[:-1]
            + f", filters={self.filters}, kernel_size={self.kernel_size}, strides={self.strides}, padding={self.padding}, activation={self.activation})"
        )

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        """The call method for the block

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor

        Returns
        -------
        (tf.keras.Model, tf.Tensor)
            The block model and the output tensor
        """
        inputs_ = inputs
        x = layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name=f"{self.name}_conv1",
            input_shape=inputs.shape,
        )(inputs)
        x = layers.BatchNormalization(name=f"{self.name}_bn1")(x)
        x = layers.Activation(self.activation, name=f"{self.name}_{self.activation}1")(
            x
        )
        x = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="same",
            name=f"{self.name}_conv2",
        )(x)
        x = layers.BatchNormalization(name=f"{self.name}_bn2")(x)
        x = layers.Activation(self.activation, name=f"{self.name}_{self.activation}2")(
            x
        )
        x = layers.Conv2D(
            filters=self.filters * self.r,
            kernel_size=1,
            strides=1,
            name=f"{self.name}_conv3",
        )(x)
        x = layers.BatchNormalization(name=f"{self.name}_bn3")(x)
        if inputs.shape[-1] - self.filters * self.r != 0:
            inputs = layers.Conv2D(
                filters=self.filters * self.r,
                kernel_size=1,
                strides=self.strides,
                name=f"{self.name}_conv4",
            )(inputs)
            inputs = layers.BatchNormalization(name=f"{self.name}_bn4")(inputs)
        x = layers.Add(name=f"{self.name}_add")([x, inputs])
        x = layers.Activation(self.activation, name=f"{self.name}_{self.activation}3")(
            x
        )
        block = Model(inputs=inputs_, outputs=x, name=self.name)
        return block, x

    def summary(self, input_shape):
        """Prints a summary of the block"""
        if input_shape[-1] != self.filters:
            print(
                "WARNING: The input shape does not match the number of filters in the block.\nSummary will work but the output shape will be wrong."
            )
            input_shape = (input_shape[0], input_shape[1], self.filters)
        inputs = layers.Input(shape=input_shape)
        block, _ = self(inputs)
        block.summary()
