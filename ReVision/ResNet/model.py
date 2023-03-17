import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, Sequential
from ReVision.blocks import Block, ResidualBottleneck, Residual


# Only the plain block is implemented here. For the other blocks, see the `ReVision.blocks` module
class ResNetPlain(Block):
    """The plain block in ResNet"""

    def __init__(self, name, filters, strides=1, activation="relu"):
        """The simple cnn block in ResNet without any residual connections.

        The block was used to create the baseline model in the paper. Padding is set to be "same".

        For others blocks related to the paper, the `Residual` and `ResidualBottleneck` blocks, see the `ReVision.blocks` module.

        Parameters
        ----------
        name : str
            Name of the block
        filters : int
            Number of filters in the block
        strides : int, optional
            Strides of the block, by default 1
        activation : str, optional
            Activation function, by default "relu"

        Returns
        -------
        None
        """
        super().__init__(name)
        self.filters = filters
        self.strides = strides
        self.activation = activation

    def __str__(self) -> str:
        return "Plain Block"

    def __repr__(self) -> str:
        return (
            super().__repr__()[:-1]
            + f", filters={self.filters}, strides={self.strides}, activation={self.activation})"
        )

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):

        x = layers.Conv2D(
            self.filters, 3, strides=1, padding="same", name=f"{self.name}_conv1"
        )(inputs)
        x = layers.BatchNormalization(name=f"{self.name}_bn1")(x)
        x = layers.Activation("relu", name=f"{self.name}_relu1")(x)
        x = layers.Conv2D(
            self.filters,
            3,
            strides=2,
            padding="same",
            name=f"{self.name}_conv2",
        )(x)
        x = layers.BatchNormalization(name=f"{self.name}_bn2")(x)
        x = layers.Activation("relu", name=f"{self.name}_relu2")(x)
        model = Model(inputs, x, name=self.name)
        return model, x


def num_to_alphabate(num):
    """Converts a number to alphabates

    Parameters
    ----------
    num : int
        Number to be converted

    Returns
    -------
    str
        Alphabate
    """
    alphabates = "abcdefghijklmnopqrstuvwxyz"
    res = ""
    while num > 0:
        res += alphabates[num % 26]
        num //= 26
    return res[::-1]


ALL_MODELS = [
    "plain",
    "resneta18",
    "resneta34",
    "resnetb18",
    "resnetb34",
    "resnetb50",
    "resnetb101",
    "resnetb152",
]


class ResNet:
    """A class to create a ResNet model
    The class implements a number of ResNet models with different options for expansion.
    """

    def __init__(
        self,
        with_preprocessing=True,
        input_shape=(224, 224, 3),
        output_shape=1000,
    ):
        """Initializes the ResNet model

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
        self.input_layer = layers.Input(shape=input_shape, name="input_layer")

    def __preprocess(self):
        """Creates the preprocessing layers for the model"""

        # TODO: Add more preprocessing layers
        input_layer = self.input_layer
        x = layers.experimental.preprocessing.Resizing(
            224, 224, name="resize", input_shape=self.input_shape
        )(input_layer)
        x = layers.Lambda(
            lambda x: x / 255.0 - 0.5,
            name="normalization",
        )(x)
        preprocess = Model(input_layer, x, name="Preprocessing")
        return preprocess, x

    def __repeat_block(
        self,
        block,
        name,
        num_blocks,
        inputs,
        downsample_first=True,
        expand=None,
        filters=None,
    ):
        """Repeats a block of layers for a given number of times

        Parameters
        ----------
        block : function
            The block of layers to repeat
        name : str
            Name of the residual block
        num_blocks : int
            Number of times to repeat the block
        inputs : tf.keras.layers
            Input to the block
        downsample_first : bool
            Whether to downsample the first block
        expand : str
            The expansion type for the block
            - None: No expansion
            - A or zero: Use zero padding
            - B or projection: Use 1x1 convolution
        Returns
        -------
        (tf.keras.Model, tf.keras.layers)
            The model and the output of the last layer
        """
        block_model = Sequential(name=name)
        arguments = {}
        arguments["filters"] = filters
        if block is Residual:
            arguments["expand"] = expand

        if downsample_first:
            arguments["strides"] = 2
            name_1 = f"{name}a"
            b, x = block(name=name_1, **arguments)(inputs)
        else:
            arguments["strides"] = 1
            name_1 = f"{name}a"
            b, x = block(name=name_1, **arguments)(inputs)
        block_model.add(b)

        for i in range(2, num_blocks + 1):
            arguments["strides"] = 1
            b, x = block(name=f"{name}{num_to_alphabate(i)}", **arguments)(x)
            block_model.add(b)
        return block_model, x

    def __build_top(self, inputs):
        """Builds the top layers of the model

        Parameters
        ----------
        inputs : tf.keras.layers
            Input to the top layers

        Returns
        -------
        (tf.keras.Model, tf.keras.layers)
            Top layers of the model and the output of the top layers
        """
        x = layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")(inputs)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Activation("relu", name="relu1")(x)
        x = layers.MaxPool2D(3, strides=2, padding="same", name="pool1")(x)

        top = Model(inputs, x, name="top")
        return top, x

    def __build_classifier(self, inputs):
        """Builds the classifier layers of the model

        Parameters
        ----------
        inputs : tf.keras.layers
            Input to the classifier layers

        Returns
        -------
        (tf.keras.Model, tf.keras.layers)
            Classifier layers of the model and the output of the classifier layers
        """
        x = layers.GlobalAveragePooling2D()(inputs)
        x = layers.Dense(self.num_classes, name="Output", activation="softmax")(x)
        classifier = Model(inputs, x, name="classifier")
        return classifier, x

    def __block_to_use(self, type):
        if type is None:
            return ResidualBottleneck
        type = type.lower()
        if type == "plain":
            return ResNetPlain
        elif type == "a" or type == "zero":
            return Residual
        elif type == "b" or type == "projection":
            return ResidualBottleneck
        else:
            raise ValueError(f"Invalid block type {type}")

    def _build_resnet(
        self,
        expand=None,
        blocks=[2, 2, 2, 2],
        name=None,
    ):
        """Builds a ResNet model with simple residual blocks

        Parameters
        ----------
        expand : str
            The type of expansion to use
        blocks : list
            Number of blocks in each stage
        name : str
            Name of the model

        Returns
        -------
        tf.keras.Model
            The ResNet model
        """
        model = Sequential(name=name)
        block = self.__block_to_use(expand)

        if self.with_preprocessing:
            preprocess, x = self.__preprocess()
            model.add(preprocess)
        else:
            x = self.input_layer
        top, x = self.__build_top(x)
        model.add(top)

        b1, x = self.__repeat_block(
            block=block,
            name="res1",
            num_blocks=blocks[0],
            downsample_first=False,
            inputs=x,
            filters=64,
            expand=expand,
        )
        model.add(b1)

        b2, x = self.__repeat_block(
            block=block,
            name="res2",
            num_blocks=blocks[1],
            downsample_first=True,
            expand=expand,
            inputs=x,
            filters=128,
        )
        model.add(b2)

        b3, x = self.__repeat_block(
            block=block,
            name="res3",
            num_blocks=blocks[2],
            downsample_first=True,
            expand=expand,
            inputs=x,
            filters=256,
        )
        model.add(b3)

        b4, x = self.__repeat_block(
            block=block,
            name="res4",
            num_blocks=blocks[3],
            downsample_first=True,
            expand=expand,
            inputs=x,
            filters=512,
        )
        model.add(b4)

        classifier, x = self.__build_classifier(x)
        model.add(classifier)

        return model

    def build_plain(self):
        """Builds the plain (non-ResNet) model

        Returns
        -------
        tf.keras.Model
            The plain model
        """
        expand = "plain"
        blocks = [2, 2, 2, 2]
        name = "ResNetA18"
        return self._build_resnet(expand, blocks, name)

    def build_resneta18(self):
        """Builds a ResNetA18 model"""
        expand = "A"
        blocks = [2, 2, 2, 2]
        name = "ResNetA18"
        return self._build_resnet(expand, blocks, name)

    def build_resneta34(self):
        """Builds a ResNetA34 model"""
        expand = "zero"
        blocks = [3, 4, 6, 3]
        name = "ResNetA34"
        return self._build_resnet(expand, blocks, name)

    def build_resnetb18(self):
        """Builds a ResNetB18 model"""
        expand = "projection"
        blocks = [2, 2, 2, 2]
        name = "ResNetB18"
        return self._build_resnet(expand, blocks, name)

    def build_resnetb34(self):
        """Builds a ResNetB34 model"""
        expand = "B"
        blocks = [3, 4, 6, 3]
        name = "ResNetB34"
        return self._build_resnet(expand, blocks, name)

    def build_resnetb50(self):
        """Builds a ResNetB50 model"""
        blocks = [3, 4, 6, 3]
        name = "ResNetB50"
        return self._build_resnet(blocks=blocks, name=name)

    def build_resnetb101(self):
        """Builds a ResNetB101 model"""
        blocks = [3, 4, 23, 3]
        name = "ResNetB101"
        return self._build_resnet(blocks=blocks, name=name)

    def build_resnetb152(self):
        """Builds a ResNetB152 model"""
        blocks = [3, 8, 36, 3]
        name = "ResNetB152"
        return self._build_resnet(blocks=blocks, name=name)

    def build(self, name):
        """Builds the model

        Parameters
        ----------
        name : str
            Name of the model to build

        Returns
        -------
        Model
            The model
        """
        if name.lower() not in ALL_MODELS:
            raise ValueError(f"Model name not found. Please choose from: {ALL_MODELS}")
        return eval(f"self.build_{name.lower()}()")
