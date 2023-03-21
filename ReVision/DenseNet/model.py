import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, Sequential
from ReVision.blocks import DenseBlock


SIZE_TO_LAYERS = {
    1: [6, 12, 24, 16],
    2: [6, 12, 32, 32],
    3: [6, 12, 48, 32],
    4: [6, 12, 64, 48],
}

SIZE_TO_FINAL_LAYERS = {
    1: 121,
    2: 169,
    3: 201,
    4: 264,
}


class DenseNet:
    """A class for the DenseNet model
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

    def __build_top(self, inputs, k):
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
        x = layers.Conv2D(2 * k, 7, strides=2, padding="same", name="conv1_s2")(inputs)
        x = layers.MaxPool2D(3, strides=2, padding="same", name="pool1_s2")(x)
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

    def __transition_layer(self, name, inputs, theta):
        """Builds the transition layer

        Parameters
        ----------
        name : str
            Name of the layer
        inputs : tf.keras.layers
            Input to the transition layer
        theta : float
            Compression factor

        Returns
        -------
        (tf.keras.Model, tf.keras.layers)
            Transition layer and the output of the transition layer
        """
        kernels = inputs.shape[-1]
        out_kernels = int(kernels * theta)
        x = layers.Conv2D(
            filters=out_kernels,
            kernel_size=1,
            strides=1,
            padding="same",
            name=f"{name}_conv1x1",
        )(inputs)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.MaxPool2D(3, strides=2, padding="same", name=f"{name}_pool")(x)
        layer = Model(inputs, x, name=name)
        return layer, x

    def __name(self, size, k, theta):
        """Returns the name of the model

        Parameters
        ----------
        size : int
            Size of the model
        k : int
            Growth rate
        theta : float
            Compression factor

        Returns
        -------
        str
            Name of the model
        """

        if size not in SIZE_TO_FINAL_LAYERS:
            raise ValueError(
                f"Size must be one of {list(SIZE_TO_FINAL_LAYERS.keys())}, got {size}"
            )
        if k < 12:
            raise ValueError(f"k must be greater than 12, got {k}")

        if theta > 1:
            raise ValueError(f"theta must be less than 1, got {theta}")
        if theta <= 0:
            raise ValueError(f"theta must be greater than 0, got {theta}")
        if theta == 1:
            name = f"DenseNetB{SIZE_TO_FINAL_LAYERS[size]}_{k}"
        else:
            name = f"DenseNetBC{SIZE_TO_FINAL_LAYERS[size]}_{k}_{theta}"
        self.name = name
        return name

    def _build(self, name, k, theta, size):
        """Builds the model given `k`

        Parameters
        ----------
        name : str
            Name of the model
        k : int
            Growth rate
        theta : float
            Compression factor
        size : int
            Size of the model

        Returns
        -------
        tf.keras.Model
            The model
        """
        model = Sequential(name=name)
        self.final_input_shape = (224, 224, 3)

        if self.with_preprocessing:
            preprocess, x = self.__preprocess()
            model.add(preprocess)
        else:
            x = self.input_layer
        top, x = self.__build_top(x, k)
        model.add(top)
        blocks = SIZE_TO_LAYERS[size]
        for i, b in enumerate(blocks):

            db = DenseBlock(name=f"denseblock_{i+1}", blocks=b, k=k)
            b_, x = db(x)
            model.add(b_)
            if i == len(blocks) - 1:
                break
            b_, x = self.__transition_layer(f"Transition_{i+1}", x, theta)
            model.add(b_)

        classifier, x = self.__build_classifier(x)
        model.add(classifier)

        return model

    def build(self, size=1, k=12, theta=0.5):
        """Builds the DenseNet model

        Parameters
        ----------
        size : int, optional
            This parameter decides which model to use:
            - 1: DenseNet-121
            - 2: DenseNet-169
            - 3: DenseNet-201
            - 4: DenseNet-264

        k : int, optional
            The number of filters in the model, by default 12
        theta : float, optional
            The compression factor, by default 1

        Returns
        -------
        tf.keras.Model
            The EfficientNet model
        """
        name = self.__name(size, k, theta)
        self.model = self._build(name, k, theta, size)
        return self.model
