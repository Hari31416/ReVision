import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, Sequential


# TODO Make it in a better way
alphabates = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
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

# TODO Change the models such that the summary is made up of smaller models
class ResNet:
    """A class to create a ResNet model
    The class implements a number of ResNet models with different options for expansion
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

    def __preprocess(self):
        """Creates the preprocessing layers for the model"""

        input_layer = layers.Input(shape=self.input_shape, name="Preprocessing_Input")
        x = layers.experimental.preprocessing.Resizing(
            224, 224, name="resize", input_shape=self.input_shape
        )(input_layer)
        x = layers.Lambda(
            lambda x: x / 255.0 - 0.5,
            name="normalization",
        )(x)
        return input_layer, x

    def plain_block(self, input, kernels, strides, name):
        """Creates a simple block for resnet"""
        x = layers.Conv2D(
            kernels, 3, strides=strides, padding="same", name=f"{name}_conv1"
        )(input)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu", name=f"{name}_relu1")(x)
        x = layers.Conv2D(
            kernels,
            3,
            strides=1,
            padding="same",
            name=f"{name}_conv2",
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.Activation("relu", name=f"{name}_relu2")(x)
        return x

    def res_block(self, input, kernels, strides, name, expand=None):
        """Creates a residual block with give options for expansion

        Parameters
        ----------
        input : tf.keras.layers.Input
            Input to the residual block
        kernels : int
            Number of kernels in the residual block
        strides : int
            Stride of the residual block
        name : str
            Name of the residual block
        expand : str, optional
            Type of expansion, by default "projection". There are two types of expansion
            - projection or B: Projection expansion
            - zero or A: Zero padding expansion

        Returns
        -------
        tf.keras.layers
            Output of the residual block

        """
        # print(f"Residual block {name} with {kernels} kernels and stride {strides}")
        # The first convolutional layers are followed by batch normalization and ReLU activation
        x = layers.Conv2D(
            kernels, 3, strides=strides, padding="same", name=f"{name}_conv1"
        )(input)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu", name=f"{name}_relu1")(x)

        # The second convolutional layers are followed by batch normalization
        x = layers.Conv2D(kernels, 3, strides=1, padding="same", name=f"{name}_conv2")(
            x
        )
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)

        if strides != 1:
            # If the stride is not 1, expand the input to match the output shape
            if expand == "projection" or expand == "B":
                # Projection expansion
                input = layers.Conv2D(
                    kernels,
                    1,
                    strides=strides,
                    padding="same",
                    name=f"{name}_projection",
                )(input)
                input = layers.BatchNormalization(name=f"{name}_bn3")(input)

            elif expand == "zero" or expand == "A":
                # Zero padding expansion
                def pad_depth(x, desired_channels):
                    """Pads the depth of the input tensor to match the desired channels"""
                    y = tf.zeros_like(x, name=f"{name}_pad_depth1")
                    new_channels = desired_channels - x.shape.as_list()[-1]
                    y = y[..., :new_channels]
                    return layers.concatenate([x, y], name=f"{name}_pad_depth2")

                desired_channels = x.shape.as_list()[-1]
                input = layers.MaxPool2D(
                    pool_size=(1, 1),
                    strides=(2, 2),
                    padding="same",
                    name=f"{name}_zero_upscale_pool",
                )(input)
                input = layers.Lambda(
                    pad_depth,
                    arguments={"desired_channels": desired_channels},
                    name=f"{name}_zero_upscale_lambda",
                )(input)
                input = layers.BatchNormalization(name=f"{name}_bn3")(input)

        # Add the input to the output
        x = layers.Add(name=f"{name}_add")([x, input])
        x = layers.Activation("relu", name=f"{name}_relu2")(x)
        return x

    def bottleneck_res_block(self, input, kernels, strides, name):
        """Creates a bottleneck residual block

        Parameters
        ----------
        input : tf.keras.layers.Input
            Input to the residual block
        kernels : int
            Number of kernels in the residual block
        strides : int
            Stride of the residual block
        name : str
            Name of the residual block

        Returns
        -------
        tf.keras.layers
            Output of the residual block
        """
        x = layers.Conv2D(kernels, 1, strides=1, padding="same", name=f"{name}_conv1")(
            input
        )
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu", name=f"{name}_relu1")(x)
        x = layers.Conv2D(
            kernels, 3, strides=strides, padding="same", name=f"{name}_conv2"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.Activation("relu", name=f"{name}_relu2")(x)
        x = layers.Conv2D(
            kernels * 4, 1, strides=1, padding="same", name=f"{name}_conv3"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn3")(x)
        if input.shape[-1] - kernels * 4 != 0:
            input = layers.Conv2D(
                kernels * 4, 1, strides=strides, padding="same", name=f"{name}_conv4"
            )(input)
            input = layers.BatchNormalization(name=f"{name}_bn4")(input)
        x = layers.Add(name=f"{name}_add")([x, input])
        x = layers.Activation("relu", name=f"{name}_relu3")(x)
        return x

    def __repeat_block(
        self,
        block,
        name,
        num_blocks,
        downsample_first=True,
        **kwargs,
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
        downsample_first : bool
            Whether to downsample the first block
        **kwargs: dict
            Arguments to pass to the block
        Returns
        -------
        tf.keras.layers
            Output of the residual block
        """
        if downsample_first:
            kwargs["strides"] = 2
            name_1 = f"{name}a"
            x = block(name=name_1, **kwargs)
        else:
            kwargs["strides"] = 1
            name_1 = f"{name}a"
            x = block(name=name_1, **kwargs)
        kwargs["input"] = x
        for i in range(2, num_blocks + 1):
            kwargs["strides"] = 1
            x = block(name=f"{name}{alphabates[i]}", **kwargs)
            kwargs["input"] = x
        return x

    def _build_resnetAB_simple(
        self,
        expand="A",
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
        if self.with_preprocessing:
            input_, x = self.__preprocess()
        else:
            input_ = layers.Input(shape=(224, 224, 3), name="Input")
            x = input_
        x = layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")(x)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Activation("relu", name="relu1")(x)
        x = layers.MaxPool2D(3, strides=2, padding="same", name="pool1")(x)
        x = self.__repeat_block(
            block=self.res_block,
            name="res1",
            num_blocks=blocks[0],
            downsample_first=False,
            input=x,
            kernels=64,
            expand=expand,
        )
        x = self.__repeat_block(
            block=self.res_block,
            name="res2",
            num_blocks=blocks[1],
            downsample_first=True,
            expand=expand,
            input=x,
            kernels=128,
        )
        x = self.__repeat_block(
            block=self.res_block,
            name="res3",
            num_blocks=blocks[1],
            downsample_first=True,
            expand=expand,
            input=x,
            kernels=256,
        )
        x = self.__repeat_block(
            block=self.res_block,
            name="res4",
            num_blocks=blocks[1],
            downsample_first=True,
            expand=expand,
            input=x,
            kernels=512,
        )

        x = layers.GlobalAveragePooling2D()(x)
        out = layers.Dense(self.num_classes, name="Output", activation="softmax")(x)
        model = Model(input_, out, name=name)
        return model

    def _build_resnetAB_bottleneck(
        self,
        blocks=[3, 4, 6, 3],
        name=None,
    ):
        """Builds a ResNet model with bottleneck residual blocks

        Parameters
        ----------
        blocks : list
            Number of blocks in each stage
        name : str
            Name of the model

        Returns
        -------
        tf.keras.Model
            The ResNet model
        """
        if self.with_preprocessing:
            input_, x = self.__preprocess()
        else:
            input_ = layers.Input(shape=(224, 224, 3), name="Input")
            x = input_
        input_ = layers.Input(shape=(224, 224, 3), name="Input")
        x = layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")(input_)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Activation("relu", name="relu1")(x)
        x = layers.MaxPool2D(3, strides=2, padding="same", name="pool1")(x)
        x = self.__repeat_block(
            block=self.bottleneck_res_block,
            name="res1",
            num_blocks=blocks[0],
            downsample_first=False,
            input=x,
            kernels=64,
            expand="B",
        )

        x = self.__repeat_block(
            block=self.bottleneck_res_block,
            name="res2",
            num_blocks=blocks[0],
            downsample_first=False,
            input=x,
            kernels=128,
            expand="B",
        )

        x = self.__repeat_block(
            block=self.bottleneck_res_block,
            name="res3",
            num_blocks=blocks[0],
            downsample_first=False,
            input=x,
            kernels=256,
            expand="B",
        )

        x = self.__repeat_block(
            block=self.bottleneck_res_block,
            name="res4",
            num_blocks=blocks[0],
            downsample_first=False,
            input=x,
            kernels=512,
            expand="B",
        )

        x = layers.GlobalAveragePooling2D()(x)
        out = layers.Dense(self.num_classes, name="Output", activation="softmax")(x)
        model = Model(input_, out, name=name)
        return model

    def build_plain(self):
        """Builds the plain (non-ResNet) model

        Returns
        -------
        tf.keras.Model
            The plain model
        """
        if self.with_preprocessing:
            input_, x = self.__preprocess()
        else:
            input_ = layers.Input(shape=(224, 224, 3), name="Input")
            x = input_
        input_ = layers.Input(shape=(224, 224, 3), name="Input")
        x = layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")(input_)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Activation("relu", name="relu1")(x)
        x = layers.MaxPool2D(3, strides=2, padding="same", name="pool1")(x)

        x = self.__repeat_block(
            block=self.plain_block,
            name="res1",
            num_blocks=2,
            downsample_first=False,
            input=x,
            kernels=64,
        )
        x = self.__repeat_block(
            block=self.plain_block,
            name="res2",
            num_blocks=2,
            downsample_first=True,
            input=x,
            kernels=128,
        )
        x = self.__repeat_block(
            block=self.plain_block,
            name="res3",
            num_blocks=2,
            downsample_first=True,
            input=x,
            kernels=256,
        )
        x = self.__repeat_block(
            block=self.plain_block,
            name="res4",
            num_blocks=2,
            downsample_first=True,
            input=x,
            kernels=512,
        )
        x = layers.GlobalAveragePooling2D()(x)
        out = layers.Dense(self.num_classes, name="Output", activation="softmax")(x)
        model = Model(input_, out, name="Plain")
        return model

    def build_resneta18(self):
        """Builds a ResNetA18 model"""
        expand = "A"
        blocks = [2, 2, 2, 2]
        name = "ResNetA18"
        return self._build_resnetAB_simple(expand, blocks, name)

    def build_resneta34(self):
        """Builds a ResNetA34 model"""
        expand = "A"
        blocks = [3, 4, 6, 3]
        name = "ResNetA34"
        return self._build_resnetAB_simple(expand, blocks, name)

    def build_resnetb18(self):
        """Builds a ResNetB18 model"""
        expand = "B"
        blocks = [2, 2, 2, 2]
        name = "ResNetB18"
        return self._build_resnetAB_simple(expand, blocks, name)

    def build_resnetb34(self):
        """Builds a ResNetB34 model"""
        expand = "B"
        blocks = [3, 4, 6, 3]
        name = "ResNetB34"
        return self._build_resnetAB_simple(expand, blocks, name)

    def build_resnetb50(self):
        """Builds a ResNetB50 model"""
        blocks = [3, 4, 6, 3]
        name = "ResNetB50"
        return self._build_resnetAB_bottleneck(blocks, name)

    def build_resnetb101(self):
        """Builds a ResNetB101 model"""
        blocks = [3, 4, 23, 3]
        name = "ResNetB101"
        return self._build_resnetAB_bottleneck(blocks, name)

    def build_resnetb152(self):
        """Builds a ResNetB152 model"""
        blocks = [3, 8, 36, 3]
        name = "ResNetB152"
        return self._build_resnetAB_bottleneck(blocks, name)

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
