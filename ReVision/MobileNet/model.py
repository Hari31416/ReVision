import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, Sequential
from ReVision.blocks import MobileV1


RHO_TO_RES = {
    "baseline": 224,
    "medium": 192,
    "small": 160,
    "tiny": 128,
}


class MobileNet:
    """A class for the MobileNet model"""

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
        self.final_input_shape = None
        self.input_layer = layers.Input(shape=input_shape, name="input_layer")

    def __preprocess(self):
        """Creates the preprocessing layers for the model"""

        # TODO: Add more preprocessing layers
        input_layer = self.input_layer
        x = layers.experimental.preprocessing.Resizing(
            self.final_input_shape[0],
            self.final_input_shape[1],
            name="resize",
            input_shape=self.input_shape,
        )(input_layer)
        x = layers.Lambda(
            lambda x: x / 255.0 - 0.5,
            name="normalization",
        )(x)
        preprocess = Model(input_layer, x, name="Preprocessing")
        return preprocess, x

    def __repeat_block(
        self,
        name,
        num_blocks,
        inputs,
        downsample_first=True,
        filters=None,
    ):
        """Repeats a block of layers for a given number of times

        Parameters
        ----------
        name : str
            Name of the residual block
        num_blocks : int
            Number of times to repeat the block
        inputs : tf.keras.layers
            Input to the block
        downsample_first : bool
            Whether to downsample the first block
        Returns
        -------
        (tf.keras.Model, tf.keras.layers)
            The model and the output of the last layer
        """
        block_model = Sequential(name=name)
        arguments = {}
        arguments["filters"] = filters
        arguments["kernel_size"] = 3

        if downsample_first:
            arguments["strides"] = 2
            name_1 = f"{name}_1"
            b, x = MobileV1(name=name_1, **arguments)(inputs)
        else:
            arguments["strides"] = 1
            name_1 = f"{name}_1"
            b, x = MobileV1(name=name_1, **arguments)(inputs)
        block_model.add(b)

        for i in range(2, num_blocks + 1):
            arguments["strides"] = 1
            b, x = MobileV1(name=f"{name}_{(i)}", **arguments)(x)
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
        x = layers.Conv2D(32, 3, strides=2, padding="same", name="conv1_s2")(inputs)
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

    def __resolution(self, rho="baseline"):
        """Calculates the resolution of image to be fed in the model

        Parameters
        ----------
        rho : str
            The resolution of the model
            Can be one of "baseline", "medium", "small", "tiny"

        Returns
        -------
        None
        """
        if rho not in RHO_TO_RES:
            raise ValueError(
                f"Invalid rho value: {rho}. Choose from {RHO_TO_RES.keys()}"
            )
        return (RHO_TO_RES[rho], RHO_TO_RES[rho], 3)

    def __filters(self, filter, alpha=1):
        """Calculates the number of filters in each layer

        Parameters
        ----------
        filter : int
            The number of filters in the baseline
        alpha : float
            The width of the model

        Returns
        -------
        None
        """
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0 ", alpha)
        if alpha <= 1:
            return int(filter * alpha)
        else:
            raise ValueError("Alpha must be less than 1", alpha)

    def __build(self, alpha=1, rho="baseline"):
        """Builds the model

        Parameters
        ----------
        name : str
            Name of the model
        alpha : float, optional
            The width of the model, by default 1
        rho : str, optional
            The resolution of the model, by default "baseline"

        Returns
        -------
        tf.keras.Model
            The model

        """
        if rho not in RHO_TO_RES:
            raise ValueError(
                f"Invalid rho value: {rho}. Choose from {RHO_TO_RES.keys()}"
            )

        name = f"{alpha}_MobileNet_{RHO_TO_RES[rho]}"
        model = Sequential(name=name)
        self.final_input_shape = self.__resolution(rho)

        if self.with_preprocessing:
            preprocess, x = self.__preprocess()
            model.add(preprocess)
        else:
            x = self.input_layer
        top, x = self.__build_top(x)
        model.add(top)

        b1, x = self.__repeat_block("Conv_1", 1, x, False, self.__filters(64, alpha))
        model.add(b1)

        b2, x = self.__repeat_block("Conv_2", 2, x, True, self.__filters(128, alpha))
        model.add(b2)

        b3, x = self.__repeat_block("Conv_3", 2, x, True, self.__filters(256, alpha))
        model.add(b3)

        b4, x = self.__repeat_block("Conv_4", 6, x, True, self.__filters(512, alpha))
        model.add(b4)

        b5, x = self.__repeat_block("Conv_5", 1, x, True, self.__filters(1024, alpha))
        model.add(b5)

        b6, x = self.__repeat_block("Conv_6", 1, x, False, self.__filters(1024, alpha))
        model.add(b6)

        classifier, x = self.__build_classifier(x)
        model.add(classifier)

        return model

    def __parse_name(self, name):
        """Parses the name of the model

        Parameters
        ----------
        name : str
            Name of the model

        Returns
        -------
        (float, str)
            The width of the model and the resolution of the model
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string")
        if len(name) < 10:
            raise ValueError("Invalid name")
        names = name.split("_")
        if len(names) != 3:
            raise ValueError("Invalid Name.")
        alpha = float(names[0])
        rho = names[2]
        return alpha, rho

    def build(self, name=None, alpha=1, rho="baseline"):
        """Builds the model

        Parameters
        ----------
        name : str
            Name of the model
        alpha : float, optional
            The width of the model, by default None (Not required if name is provided)
        rho : str, optional
            The resolution of the model, by default "baseline" (Not required if name is provided)

        Returns
        -------
        tf.keras.Model
            The model

        """
        if name is not None:
            alpha, rho = self.__parse_name(name)
        self.model = self.__build(alpha, rho)
        return self.model
