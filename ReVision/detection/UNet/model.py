import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential, Model


class UNet:
    """UNet model for image segmentation"""

    def __init__(
        self,
        with_preprocessing=True,
        input_shape=(512, 512, 3),
        output_classes=1,
    ) -> None:
        """Initializes the UNet model

        Parameters
        ----------
        with_preprocessing : bool, optional
            Whether to include the preprocessing layers, by default True
        input_shape : tuple, optional
            Shape of the input, by default (512, 512, 3)
        output_classes : int, optional
            Number of output classes, by default 1
        """
        self.with_preprocessing = with_preprocessing
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.input_layer = layers.Input(shape=input_shape, name="input_layer")

    def __preprocess(self):
        """Creates the preprocessing layers for the model"""

        input_layer = self.input_layer
        x = layers.experimental.preprocessing.Resizing(
            self.input_shape[0],
            self.input_shape[1],
            name="resize",
            input_shape=self.input_shape,
        )(input_layer)
        x = layers.Lambda(
            lambda x: x / 255.0,
            name="normalization",
        )(x)
        preprocess = Model(input_layer, x, name="Preprocessing")
        return preprocess, x

    def _cotracting_block(self, inputs, filters, kernels, name):
        """Creates a contracting block

        Parameters
        ----------
        inputs : tf.keras.layers
            Input to the contracting block
        filters : int
            Number of filters in the convolutional layers
        kernels : int
            Kernel size of the convolutional layers
        name : str
            Name of the block

        Returns
        -------
        (tf.keras.layers, tf.keras.layers)
            Output of the contracting block and the output of the max pooling layer
        """
        x = layers.Conv2D(
            filters,
            kernels,
            activation="relu",
            padding="same",
            name=f"{name}_conv1",
        )(inputs)
        x = layers.Conv2D(
            filters,
            kernels,
            activation="relu",
            padding="same",
            name=f"{name}_conv2",
        )(x)
        skip = x
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name=f"{name}_pool")(x)
        return x, skip

    def _expansive_block(self, inputs, skip, filters, kernels, name):
        """Creates an expansive block

        Parameters
        ----------
        inputs : tf.keras.layers
            Input to the expansive block
        skip : tf.keras.layers
            Input to the expansive block from the contracting block
        filters : int
            Number of filters in the convolutional layers
        kernels : int
            Kernel size of the convolutional layers
        name : str
            Name of the block

        Returns
        -------
        tf.keras.layers
            Output of the expansive block
        """
        x = layers.UpSampling2D(size=(2, 2), name=f"{name}_upsample")(inputs)
        x = layers.Concatenate(name=f"{name}_concat")([x, skip])
        x = layers.Conv2DTranspose(
            filters,
            kernels,
            activation="relu",
            padding="same",
            name=f"{name}_conv1",
        )(x)
        x = layers.Conv2DTranspose(
            filters,
            kernels,
            activation="relu",
            padding="same",
            name=f"{name}_conv2",
        )(x)
        return x

    def _bottleneck(self, inputs, filters, kernels, name):
        """Creates a bottleneck block

        Parameters
        ----------
        inputs : tf.keras.layers
            Input to the bottleneck block
        filters : int
            Number of filters in the convolutional layers
        kernels : int
            Kernel size of the convolutional layers
        name : str
            Name of the block

        Returns
        -------
        tf.keras.layers
            Output of the bottleneck block
        """
        x = layers.Conv2D(
            filters,
            kernels,
            activation="relu",
            padding="same",
            name=f"{name}_conv1",
        )(inputs)
        x = layers.Conv2D(
            filters,
            kernels,
            activation="relu",
            padding="same",
            name=f"{name}_conv2",
        )(x)
        return x

    def _output_block(self, inputs, name):
        """Creates an output block

        Parameters
        ----------
        inputs : tf.keras.layers
            Input to the output block
        name : str
            Name of the block

        Returns
        -------
        tf.keras.layers
            Output of the output block
        """
        x = layers.Conv2D(
            self.output_classes,
            1,
            activation="sigmoid",
            padding="same",
            name=f"{name}_1x1",
        )(inputs)

        return x

    def build(self):
        """Builds the UNet model"""
        filters = [64, 128, 256, 512]
        bottleneck_filters = 1024
        skips = []

        if self.with_preprocessing:
            _, x = self.__preprocess()
        else:
            x = self.input_layer

        for i in range(len(filters)):
            x, skip = self._cotracting_block(x, filters[i], 3, f"contract_{i+1}")
            skips.append(skip)
        skips = skips[::-1]
        filters = filters[::-1]
        x = self._bottleneck(x, bottleneck_filters, 3, "bottleneck")

        for i in range(len(filters)):
            x = self._expansive_block(x, skips[i], filters[i], 3, f"expand_{i+1}")
        x = self._output_block(x, "output")

        model = Model(self.input_layer, x, name="UNet")
        return model
