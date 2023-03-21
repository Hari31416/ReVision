import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential


class Block:
    """Base class for all blocks"""

    def __init__(self, name):
        self.name = name
        self.filters = None

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(name={self.name})"

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        raise NotImplementedError

    def summary(self, input_shape):
        """Prints a summary of the block"""
        if input_shape[-1] != self.filters:
            print(
                "WARNING: The input shape does not match the number of filters in the block.\nSummary will work but the output shape will be wrong."
            )
            input_shape = (input_shape[0], input_shape[1], self.filters)
        inputs = layers.Input(shape=input_shape, name=self.name + "_input")
        block, _ = self(inputs)
        block.summary()


class InceptionNaive(Block):
    """This is the implementation of the naive inception block (no projection)"""

    def __init__(
        self,
        name,
        filters,
        activation="relu",
    ):
        """Initializes the block

        Parameters
        ----------
        name : str
            Name of the block
        filters : list
            List of filters for the 1x1, 3x3, 5x5 and max pooling layers
        activation : str
            Activation function to use

        Raises
        ------
        ValueError
            If the number of filters is not 3
        """
        super().__init__(name)

        if not isinstance(filters, list):
            raise ValueError("filters must be a list")
        if len(filters) != 3:
            raise ValueError("filters must have 3 elements")
        self.filters = filters

        self.activation = activation

    def __str__(self) -> str:
        return "Inception Block"

    def __repr__(self) -> str:
        return (
            super().__repr__()[:-1]
            + f", filters={self.filters}, activation={self.activation})"
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
        kernels = self.filters
        if len(kernels) != 3:
            raise ValueError("kernels must have 3 elements")
        # 1x1 conv
        x1 = layers.Conv2D(
            kernels[0],
            (1, 1),
            padding="same",
            activation=self.activation,
            name=f"{self.name}_1x1",
        )(inputs)

        # 3x3 conv
        x3 = layers.Conv2D(
            kernels[1],
            (3, 3),
            padding="same",
            activation=self.activation,
            name=f"{self.name}_3x3",
        )(inputs)

        # 5x5 conv
        x5 = layers.Conv2D(
            kernels[2],
            (5, 5),
            padding="same",
            activation=self.activation,
            name=f"{self.name}_5x5",
        )(inputs)

        # 3x3 max pooling
        x_pool = layers.MaxPooling2D(
            (3, 3), strides=(1, 1), padding="same", name=f"{self.name}_pool"
        )(inputs)

        # concatenate filters, assumes filters/channels last
        x = layers.concatenate(
            [x1, x3, x5, x_pool], axis=-1, name=f"{self.name}_concat"
        )

        # define model
        model = Model(inputs=inputs, outputs=x, name=self.name)
        return model, x

    def summary(self, input_shape):
        """Prints a summary of the block"""
        inputs = layers.Input(shape=input_shape, name=self.name + "_input")
        block, _ = self(inputs)
        block.summary()


class InceptionBottleneck(Block):
    """This is the implementation of the inception block (with projection)"""

    def __init__(
        self,
        name,
        filters,
        projection_filters,
        activation="relu",
    ):
        """The constructor for the inception block

        Parameters
        ----------
        name : str
            The name of the block
        filters : list
            The number of filters for the 1x1, 3x3, and 5x5 convolutions
        projection_filters : list
            The number of filters for the 1x1 convolutions before the 3x3 and 5x5 convolutions and the 3x3 max pooling
        activation : str
            The activation function to use

        Raises
        ------
        ValueError
            If `filters` or `projection_filters` are not lists or do not have the correct number of elements
        """
        super().__init__(name)

        if not isinstance(filters, list):
            raise ValueError("filters must be a list")
        if len(filters) != 3:
            raise ValueError("filters must have 3 elements")
        self.filters = filters

        if not isinstance(projection_filters, list):
            raise ValueError("projection_filters must be a list")
        if len(projection_filters) != 3:
            raise ValueError("projection_filters must have 2 elements")
        self.projection_filters = projection_filters

        self.activation = activation

    def __str__(self) -> str:
        return "Inception Block"

    def __repr__(self) -> str:
        return (
            super().__repr__()[:-1]
            + f", filters={self.filters}, activation={self.activation}, projection_filters={self.projection_filters})"
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
        # 1x1 conv
        x1 = layers.Conv2D(
            self.filters[0],
            (1, 1),
            padding="same",
            activation=self.activation,
            name=f"{self.name}_1x1",
        )(inputs)

        # 3x3 conv
        x_temp = layers.Conv2D(
            self.projection_filters[0],
            (1, 1),
            padding="same",
            activation=self.activation,
            name=f"{self.name}_3x3_1x1",
        )(inputs)
        x3 = layers.Conv2D(
            self.filters[1],
            (3, 3),
            padding="same",
            activation=self.activation,
            name=f"{self.name}_3x3",
        )(x_temp)

        # 5x5 conv
        x_temp = layers.Conv2D(
            self.projection_filters[1],
            (1, 1),
            padding="same",
            activation=self.activation,
            name=f"{self.name}_5x5_1x1",
        )(inputs)
        x5 = layers.Conv2D(
            self.filters[2],
            (5, 5),
            padding="same",
            activation=self.activation,
            name=f"{self.name}_5x5",
        )(x_temp)

        # 3x3 max pooling
        x_pool = layers.MaxPooling2D(
            (3, 3), strides=(1, 1), padding="same", name=f"{self.name}_pool"
        )(inputs)
        x_pool = layers.Conv2D(
            self.projection_filters[2],
            (1, 1),
            padding="same",
            activation=self.activation,
            name=f"{self.name}_pool_1x1",
        )(x_pool)

        # concatenate filters, assumes filters/channels last
        x = layers.concatenate(
            [x1, x3, x5, x_pool], axis=-1, name=f"{self.name}_concat"
        )

        # define model
        model = Model(inputs=inputs, outputs=x, name=self.name)
        return model, x

    def summary(self, input_shape):
        """Prints a summary of the block"""
        inputs = layers.Input(shape=input_shape, name=self.name + "_input")
        block, _ = self(inputs)
        block.summary()


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


class MobileV1(Block):
    """The Mobile V1 block.

    This is the block used in the original MobileNet paper.
    """

    def __init__(
        self,
        name,
        kernel_size,
        filters,
        strides,
        activation="relu",
    ):
        """The Mobile V1 block

        Parameters
        ----------
        name : str
            The name of the block
        kernel_size : int
            The kernel size of the block
        filters : int
            The number of filters in the block
        strides : int
            The stride of the block
        activation : str, optional
            The activation function of the block, by default "relu"

        """
        super().__init__(name)
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.activation = activation

    def __str__(self) -> str:
        return "Mobile V1 Block"

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
        x = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="same",
            name=f"{self.name}_dw_s{self.strides}",
        )(inputs)
        x = layers.BatchNormalization(name=f"{self.name}_dw_bn")(x)
        x = layers.Activation(
            self.activation, name=f"{self.name}_dw_{self.activation}"
        )(x)
        x = layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name=f"{self.name}_1x1",
        )(x)
        x = layers.BatchNormalization(name=f"{self.name}_1x1_bn")(x)
        x = layers.Activation(self.activation, name=f"{self.name}_{self.activation}")(x)
        block = Model(inputs=inputs_, outputs=x, name=self.name)
        return block, x


class SE(Block):
    """The squeeze and excitation block. This is the block used in the SENet architecture.

    The block was also used in EfficientNet with the MBConv block.
    """

    def __init__(
        self,
        name,
        filters,
        ratio=1 / 16,
        activation="relu",
    ):
        """The squeeze and excitation block

        Parameters
        ----------
        name : str
            The name of the block
        filters : int
            The number of filters in the block
        ratio : int, optional
            The ratio of the squeeze and excitation block, by default 1/16
        activation : str, optional
            The activation function of the block, by default "relu"
        """
        super().__init__(name)
        self.filters = filters
        self.activation = activation
        if ratio > 1:
            raise ValueError("The ratio must be less than 1")
        self.r = ratio

    def __str__(self) -> str:
        return "Squeeze and Excitation Block"

    def __repr__(self) -> str:
        return (
            super().__repr__()[:-1]
            + f", filters={self.filters}, ratio={self.r}, activation={self.activation})"
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
        x = layers.GlobalAveragePooling2D(name=f"{self.name}_gpool")(inputs)
        x = layers.Dense(
            units=int(self.filters * self.r),
            activation=self.activation,
            name=f"{self.name}_dense1",
        )(x)
        x = layers.Dense(
            units=self.filters,
            activation="sigmoid",
            name=f"{self.name}_dense2",
        )(x)
        x = layers.Multiply(name=f"{self.name}_multiply")([inputs, x])
        block = Model(inputs=inputs_, outputs=x, name=self.name)
        return block, x


class MBConv(Block):
    """The MBConv block. This is the block used in the MobileNet V2 paper.

    This implementations also adds the squeeze and excitation block. Thus it can be used in EfficientNet.
    """

    def __init__(
        self,
        name,
        kernel_size,
        filters,
        strides,
        expansion_factor,
        activation="relu",
        se_ratio=0,
    ):
        """The MBConv block

        Parameters
        ----------
        name : str
            The name of the block
        kernel_size : int
            The kernel size of the block
        filters : int
            The number of filters in the block
        strides : int
            The stride of the block
        expansion_factor : int
            The expansion factor of the block
        activation : str, optional
            The activation function of the block, by default "relu"
        se_ratio : int, optional
            The squeeze and excitation ratio, by default 0
        """
        super().__init__(name)
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.expansion_factor = expansion_factor
        self.activation = activation
        self.se_ratio = se_ratio

    def __str__(self) -> str:
        return "MBConv Block"

    def __repr__(self) -> str:
        return (
            super().__repr__()[:-1]
            + f", filters={self.filters}, kernel_size={self.kernel_size}, strides={self.strides}, expansion_factor={self.expansion_factor}, activation={self.activation}, se_ratio={self.se_ratio})"
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
            filters=self.filters * self.expansion_factor,
            kernel_size=1,
            strides=1,
            padding="same",
            name=f"{self.name}_conv1",
        )(inputs)
        x = layers.BatchNormalization(name=f"{self.name}_bn1")(x)
        x = layers.Activation(self.activation, name=f"{self.name}_{self.activation}1")(
            x
        )
        x = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="same",
            name=f"{self.name}_dw_s{self.strides}",
        )(x)
        x = layers.BatchNormalization(name=f"{self.name}_dw_bn")(x)
        x = layers.Activation(
            self.activation, name=f"{self.name}_dw_{self.activation}"
        )(x)
        if self.se_ratio > 0:
            x = SE(
                name=f"{self.name}_se",
                filters=self.filters * self.expansion_factor,
                ratio=self.se_ratio,
                activation=self.activation,
            )(x)[1]

        x = layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name=f"{self.name}_conv2",
        )(x)
        x = layers.BatchNormalization(name=f"{self.name}_bn2")(x)
        if self.strides == 1 and self.filters == inputs.shape[-1]:
            x = layers.Add(name=f"{self.name}_add")([inputs, x])
        else:
            inputs = layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                strides=self.strides,
                padding="same",
                name=f"{self.name}_conv3",
            )(inputs_)
            inputs = layers.BatchNormalization(name=f"{self.name}_bn3")(inputs)
            x = layers.Add(name=f"{self.name}_add")([inputs, x])
        block = Model(inputs=inputs_, outputs=x, name=self.name)
        return block, x


class DenseBlock(Block):
    """The dense block used in DenseNet"""

    def __init__(
        self,
        name,
        blocks,
        k=12,
        activation="relu",
    ):
        """The dense block"""
        super().__init__(name)
        self.blocks = blocks
        self.k = k
        self.activation = activation

    def __str__(self) -> str:
        return "Dense Block"

    def __repr__(self) -> str:
        return (
            super().__repr__()[:-1]
            + f", blocks={self.blocks}, k={self.k}, activation={self.activation})"
        )

    def __call__(self, inputs):
        return self.call(inputs)

    def __block(self, name, x, inputs_to_concat):
        """Creates the sub block for the dense block

        Parameters
        ----------
        name : str
            The name of the block
        x : tf.Tensor
            The input tensor
        inputs_to_concat : list
            The list of tensors to concatenate

        Returns
        -------
        (tf.Tensor, list)
            The output tensor and the list of tensors to concatenate
        """
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation(self.activation, name=f"{name}_{self.activation}1")(x)
        x = layers.Conv2D(
            filters=4 * self.k,
            kernel_size=1,
            strides=1,
            padding="same",
            name=f"{name}_conv1x1",
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.Activation(self.activation, name=f"{name}_{self.activation}2")(x)
        x = layers.Conv2D(
            filters=self.k,
            kernel_size=3,
            strides=1,
            padding="same",
            name=f"{name}_conv3x3",
        )(x)
        inputs_to_concat.append(x)
        x = layers.Concatenate(name=f"{name}_concat")(inputs_to_concat)
        return x, inputs_to_concat

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
        x = inputs
        inputs_to_concat = [x]
        for i in range(self.blocks):
            x, inputs_to_concat = self.__block(
                name=f"{self.name}_{i+1}", x=x, inputs_to_concat=inputs_to_concat
            )

        x = layers.BatchNormalization(name=f"{self.name}_bn")(x)
        x = layers.Activation(self.activation, name=f"{self.name}_{self.activation}")(x)

        block = Model(inputs=inputs_, outputs=x, name=self.name)
        return block, x

    def summary(self, input_shape):
        """Prints a summary of the block"""
        inputs = layers.Input(shape=input_shape, name=self.name + "_input")
        block, _ = self(inputs)
        block.summary()
