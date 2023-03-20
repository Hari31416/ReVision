import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, Sequential
from ReVision.blocks import MBConv

BASELINE_FILTER = [16, 24, 40, 80, 112, 192, 320]  # filters in each layer
BASELINE_LAYERS = [1, 2, 2, 3, 3, 4, 1]  # number of layers in each block
DOWNSAMPLE = [
    False,
    True,
    True,
    True,
    False,
    True,
    False,
]  # whether to downsample in the particular blocks
EXPANSION_RATIOS = [1, 6, 6, 6, 6, 6, 6]  # expansion ratios in each block
KERNELS = [3, 3, 5, 3, 5, 5, 3]  # kernel sizes in each block
SE_RATIO = 1 / 16  # squeeze and excitation ratio

ALPHA = 1.2  # depth multiplier
BETA = 1.1  # width multiplier
GAMMA = 1.15  # resolution multiplier

PHI_TO_RES = {
    0: 224,
    1: 240,
    2: 260,
    3: 300,
    4: 380,
    5: 456,
    6: 528,
    7: 600,
}  # resolution of the model for each phi value


class EfficientNet:
    """A class for the EfficientNet model
    The class has methods to implement the baseline and other variants of the EfficientNet model.

    use `build` to build the model passing either the phi value or the name of the model
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
        expansion_factor=None,
        se_ratio=None,
        kernel_size=None,
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
        filters : int
            Number of filters in the block
        expansion_factor : int
            Expansion factor of the block
        se_ratio : float
            Squeeze and Excitation ratio of the block
        kernel_size : int
            Kernel size of the block
        Returns
        -------
        (tf.keras.Model, tf.keras.layers)
            The model and the output of the last layer
        """
        block_model = Sequential(name=name)
        arguments = {}
        arguments["filters"] = filters
        arguments["kernel_size"] = kernel_size
        arguments["expansion_factor"] = expansion_factor
        arguments["se_ratio"] = se_ratio

        if downsample_first:
            arguments["strides"] = 2
            name_1 = f"{name}_1"
            b, x = MBConv(name=name_1, **arguments)(inputs)
        else:
            arguments["strides"] = 1
            name_1 = f"{name}_1"
            b, x = MBConv(name=name_1, **arguments)(inputs)
        block_model.add(b)

        for i in range(2, num_blocks + 1):
            arguments["strides"] = 1
            b, x = MBConv(name=f"{name}_{(i)}", **arguments)(x)
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

    def __parse_name(self, name):
        """Parses the name of the model

        Parameters
        ----------
        name : str
            Name of the model

        Returns
        -------
        float
            phi value
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string")
        if len(name) < 10:
            raise ValueError("Invalid name")
        try:
            phi = int(name[-1])
        except ValueError:
            raise ValueError("Invalid name")
        return phi

    def __filters(self, phi=0):
        """Calculates the number of filters in each layer"""
        filters_ = BASELINE_FILTER
        mulitplier = BETA**phi
        if phi > 0:
            filters_ = [int(f * mulitplier) for f in BASELINE_FILTER]
        return filters_

    def __layers(self, phi=0):
        """Calculates the number of layers in each block"""
        layers_ = BASELINE_LAYERS
        mulitplier = ALPHA**phi
        if phi > 0:
            layers_ = [int(l * mulitplier) for l in BASELINE_LAYERS]
        return layers_

    def __resolution(self, phi=0):
        """Calculates the resolution of image to be fed in the model"""
        return (PHI_TO_RES[phi], PHI_TO_RES[phi], 3)

    def _build(self, phi=0):
        """Builds the model given `phi`

        Parameters
        ----------
        phi : int, optional
            The resolution of the model, by default 0

        Returns
        -------
        tf.keras.Model
            The model
        """
        if phi not in PHI_TO_RES:
            raise ValueError(
                f"Invalid phi value: {phi}. Choose from {list(PHI_TO_RES.keys())}"
            )
        name = f"EfficientNetB{phi}"
        model = Sequential(name=name)
        self.final_input_shape = self.__resolution(phi)

        if self.with_preprocessing:
            preprocess, x = self.__preprocess()
            model.add(preprocess)
        else:
            x = self.input_layer
        top, x = self.__build_top(x)
        model.add(top)

        filters = self.__filters(phi)
        layers_ = self.__layers(phi)
        for i in range(len(filters)):
            b, x = self.__repeat_block(
                name=f"MBConv{i}e{EXPANSION_RATIOS[i]}k{KERNELS[i]}",
                num_blocks=layers_[i],
                inputs=x,
                filters=filters[i],
                downsample_first=DOWNSAMPLE[i],
                expansion_factor=EXPANSION_RATIOS[i],
                kernel_size=KERNELS[i],
                se_ratio=SE_RATIO,
            )
            model.add(b)

        classifier, x = self.__build_classifier(x)
        model.add(classifier)

        return model

    def build(self, name=None, phi=0):
        """Builds the EfficientNet model

        Parameters
        ----------
        name : str, optional
            Name of the model, by default None
            Name should be of the form EfficientNetB{phi}
        phi : int, optional
            The resolution of the model, by default 0
            Either 0, 1, 2, 3, 4, 5, 6, 7
            One must provide either name or phi

        Returns
        -------
        tf.keras.Model
            The EfficientNet model
        """
        if name is not None:
            phi = self.__parse_name(name)
        self.model = self._build(phi)
        return self.model
