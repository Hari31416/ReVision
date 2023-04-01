import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, Sequential


class Inception:
    """A module to implement the Inception Module."""

    def __init__(
        self,
        with_preprocessing=True,
        input_shape=(224, 224, 3),
        output_shape=1000,
    ):
        """The Inception Architecture

        Parameters
        ----------
        with_preprocessing : bool, optional
            Whether to include the preprocessing layers, by default True
        input_shape : tuple, optional
            Shape of the input, by default (224, 224, 3)
        output_shape : int, optional
            Number of classes, by default 1000

        Returns
        -------
        None
        """
        self.with_preprocessing = with_preprocessing
        self.input_shape = input_shape
        self.num_classes = output_shape

    def __preprocess(self, input):
        """Creates the preprocessing layers for the model"""

        x = layers.experimental.preprocessing.Resizing(
            224, 224, name="resize", input_shape=self.input_shape
        )(input)
        x = layers.Lambda(
            lambda x: x / 255.0 - 0.5,
            name="normalization",
        )(x)
        preprocess = Model(input, x, name="preprocess")
        return preprocess

    def inception_block_naive(
        self,
        input,
        name,
        kernels,
    ):
        """Implementation of the Naive Inception Module

        Parameters
        ----------
        input : tf.keras.layers.Input
            Input to the inception module
        name : str
            Name of the inception module
        kernels : list
            List of kernels for the 1x1, 3x3 and 5x5 convolutions

        Returns
        -------
        tf.keras.models.Model
            The inception module
        """
        if len(kernels) != 3:
            raise ValueError("kernels must have 3 elements")
        # 1x1 conv
        x1 = layers.Conv2D(
            kernels[0], (1, 1), padding="same", activation="relu", name=f"{name}_1x1"
        )(input)

        # 3x3 conv
        x3 = layers.Conv2D(
            kernels[1], (3, 3), padding="same", activation="relu", name=f"{name}_3x3"
        )(input)

        # 5x5 conv
        x5 = layers.Conv2D(
            kernels[2], (5, 5), padding="same", activation="relu", name=f"{name}_5x5"
        )(input)

        # 3x3 max pooling
        x_pool = layers.MaxPooling2D(
            (3, 3), strides=(1, 1), padding="same", name=f"{name}_pool"
        )(input)

        # concatenate filters, assumes filters/channels last
        x = layers.concatenate([x1, x3, x5, x_pool], axis=-1, name=f"{name}_concat")

        # define model
        model = Model(inputs=input, outputs=x, name=name)
        return model

    def inception_block(
        self,
        input,
        name,
        kernels,
        project_kernels,
        input_shape=None,
    ):
        """Implementation of the Naive Inception Module

        Parameters
        ----------
        input : tf.keras.layers.Input
            Input to the inception module
        name : str
            Name of the inception module
        kernels : list
            List of kernels for the 1x1, 3x3 and 5x5 convolutions
        project_kernels : list
            List of kernels for the 1x1 convolutions before the 3x3 and 5x5 convolutions

        Returns
        -------
        tf.keras.models.Model
            The inception module
        """
        # 1x1 conv
        x1 = layers.Conv2D(
            kernels[0],
            (1, 1),
            padding="same",
            activation="relu",
            name=f"{name}_1x1",
            input_shape=input_shape,
        )(input)

        # 3x3 conv
        x_temp = layers.Conv2D(
            project_kernels[0],
            (1, 1),
            padding="same",
            activation="relu",
            name=f"{name}_3x3_1x1",
        )(input)
        x3 = layers.Conv2D(
            kernels[1], (3, 3), padding="same", activation="relu", name=f"{name}_3x3"
        )(x_temp)

        # 5x5 conv
        x_temp = layers.Conv2D(
            project_kernels[1],
            (1, 1),
            padding="same",
            activation="relu",
            name=f"{name}_5x5_1x1",
        )(input)
        x5 = layers.Conv2D(
            kernels[2], (5, 5), padding="same", activation="relu", name=f"{name}_5x5"
        )(x_temp)

        # 3x3 max pooling
        x_pool = layers.MaxPooling2D(
            (3, 3), strides=(1, 1), padding="same", name=f"{name}_pool"
        )(input)
        x_pool = layers.Conv2D(
            project_kernels[2],
            (1, 1),
            padding="same",
            activation="relu",
            name=f"{name}_pool_1x1",
        )(x_pool)

        # concatenate filters, assumes filters/channels last
        x = layers.concatenate([x1, x3, x5, x_pool], axis=-1, name=f"{name}_concat")

        # define model
        model = Model(inputs=input, outputs=x, name=name)
        return model

    def simple_layer(self, inputs):
        """This creates the simple layer before the inception module

        Parameters
        ----------
        inputs : tf.keras.layers.Input
            Input to the simple layer

        Returns
        -------
        tf.keras.models.Model
            The simple layer
        """

        x = layers.Conv2D(
            64,
            (7, 7),
            strides=(2, 2),
            padding="same",
            activation="relu",
            name="conv1",
            input_shape=(224, 224, 3),
        )(inputs)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        x = layers.Conv2D(
            192, (3, 3), strides=(1, 1), padding="same", activation="relu", name="conv2"
        )(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool2")(x)

        model = Model(inputs=inputs, outputs=x, name="simple_layer")
        return model

    def classifier(self, inputs):

        x = layers.GlobalAveragePooling2D(
            name="avg_pool",
            input_shape=(7, 7, 1024),
        )(inputs)
        # x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(1000, activation="softmax", name="fc")(x)

        model = Model(inputs=inputs, outputs=x, name="classifier")
        return model

    def build(self):
        """Builds the model"""

        # TODO Make this function more modular?
        model = Sequential(name="Inception")
        inputs = layers.Input(shape=self.input_shape, name="input")
        if self.with_preprocessing:
            preprocess = self.__preprocess(inputs)
            model.add(preprocess)
            x = preprocess.output
        else:
            x = inputs

        simple = self.simple_layer(x)
        model.add(simple)

        kernels = [64, 128, 32]
        project_kernels = [96, 16, 32]
        inception3a = self.inception_block(
            simple.output,
            "inception_3a",
            kernels,
            project_kernels,
            input_shape=(28, 28, 192),
        )
        model.add(inception3a)

        kernels = [128, 192, 96]
        project_kernels = [128, 32, 64]
        inception3b = self.inception_block(
            inception3a.output,
            "inception_3b",
            kernels,
            project_kernels,
            input_shape=(28, 28, 256),
        )
        model.add(inception3b)

        x = layers.MaxPooling2D(
            (3, 3),
            strides=(2, 2),
            padding="same",
            name="pool3",
        )(inception3b.output)

        kernels = [192, 208, 48]
        project_kernels = [96, 16, 64]
        inception4a = self.inception_block(
            x,
            "inception_4a",
            kernels,
            project_kernels,
            input_shape=(14, 14, 480),
        )
        model.add(inception4a)

        kernels = [160, 224, 64]
        project_kernels = [112, 24, 64]
        inception4b = self.inception_block(
            inception4a.output,
            "inception_4b",
            kernels,
            project_kernels,
            input_shape=(14, 14, 512),
        )
        model.add(inception4b)

        kernels = [128, 256, 64]
        project_kernels = [128, 24, 64]
        inception4c = self.inception_block(
            inception4b.output,
            "inception_4c",
            kernels,
            project_kernels,
            input_shape=(14, 14, 512),
        )
        model.add(inception4c)

        kernels = [112, 288, 64]
        project_kernels = [144, 32, 64]
        inception4d = self.inception_block(
            inception4c.output,
            "inception_4d",
            kernels,
            project_kernels,
            input_shape=(14, 14, 528),
        )
        model.add(inception4d)

        kernels = [256, 320, 128]
        project_kernels = [160, 32, 128]
        inception4e = self.inception_block(
            inception4d.output,
            "inception_4e",
            kernels,
            project_kernels,
            input_shape=(14, 14, 832),
        )
        model.add(inception4e)

        x = layers.MaxPooling2D(
            (3, 3),
            strides=(2, 2),
            padding="same",
            name="pool4",
        )(inception4e.output)

        kernels = [256, 320, 128]
        project_kernels = [160, 32, 128]
        inception5a = self.inception_block(
            x,
            "inception_5a",
            kernels,
            project_kernels,
            input_shape=(7, 7, 832),
        )
        model.add(inception5a)

        kernels = [384, 384, 128]
        project_kernels = [192, 48, 128]
        inception5b = self.inception_block(
            inception5a.output,
            "inception_5b",
            kernels,
            project_kernels,
            input_shape=(7, 7, 1024),
        )
        model.add(inception5b)

        classifier = self.classifier(inception5b.output)
        model.add(classifier)

        return model
