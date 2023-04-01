import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential


class VGG:
    """This class has implementations of various models under VGG. The models are:
    - A
    - B
    - C
    - D
    - E
    """

    def __init__(
        self,
        model="A",
        with_preprocessing=True,
        input_shape=(224, 224, 3),
        output_shape=1000,
    ) -> None:
        self.model = model
        self.with_preprocessing = with_preprocessing
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __preprocess(self):
        preprocessing = Sequential(name="preprocessing")
        resize = layers.experimental.preprocessing.Resizing(
            224, 224, name="resize", input_shape=self.input_shape
        )
        normalization = layers.Lambda(
            lambda x: x / 255.0, name="normalization", input_shape=self.input_shape
        )

        preprocessing.add(resize)
        preprocessing.add(normalization)
        return preprocessing

    def __conv_layers(
        self, k=3, filters=64, numbers=1, name=None, maxpool=True, layer=0
    ):
        conv_layer = Sequential(name=name)
        for i in range(numbers):
            conv_layer.add(
                layers.Conv2D(
                    filters,
                    (k, k),
                    activation="relu",
                    name=f"{name}_{i+1}",
                    strides=(1, 1),
                    padding="same",
                )
            )
        if maxpool:
            conv_layer.add(layers.MaxPool2D((2, 2), (2, 2), name=f"maxpool_{layer}"))
        return conv_layer

    def __fc_layer(self):
        fc_layer = Sequential(name="FC")
        fc_layer.add(layers.Flatten(name="Flatten"))
        fc_layer.add(layers.Dense(4096, activation="relu", name="FC_1"))
        fc_layer.add(layers.BatchNormalization(name="BatchNorm_1"))
        fc_layer.add(layers.Dense(4096, activation="relu", name="FC_2"))
        fc_layer.add(layers.BatchNormalization(name="BatchNorm_2"))
        fc_layer.add(layers.Dense(self.output_shape, activation="relu", name="FC_3"))
        return fc_layer

    def _build_A(self):
        ks = [3, 3, 3, 3, 3]
        n_filters = [64, 128, 256, 512, 512]
        numbers = [1, 1, 2, 2, 2]
        pool_too = [True, True, True, True, True]
        model = Sequential(name="VGG-A")

        input_ = layers.Input(shape=self.input_shape, name="Input")
        model.add(input_)
        if self.with_preprocessing:
            model.add(self.__preprocess())
        for i in range(len(ks)):
            model.add(
                self.__conv_layers(
                    k=ks[i],
                    filters=n_filters[i],
                    numbers=numbers[i],
                    name=f"Conv_{i+1}",
                    maxpool=pool_too[i],
                    layer=i + 1,
                )
            )

        model.add(self.__fc_layer())
        return model

    def _build_B(self):
        ks = [3, 3, 3, 3, 3]
        n_filters = [64, 128, 256, 512, 512]
        numbers = [2, 2, 2, 2, 2]
        pool_too = [True, True, True, True, True]
        model = Sequential(name="VGG-B")
        input_ = layers.Input(shape=self.input_shape, name="Input")
        model.add(input_)
        if self.with_preprocessing:
            model.add(self.__preprocess())
        for i in range(len(ks)):
            model.add(
                self.__conv_layers(
                    k=ks[i],
                    filters=n_filters[i],
                    numbers=numbers[i],
                    name=f"Conv_{i+1}",
                    maxpool=pool_too[i],
                    layer=i + 1,
                )
            )

        model.add(self.__fc_layer())
        return model

    def _build_C(self):
        ks = [3, 3, 3, 1, 3, 1, 3, 1]
        n_filters = [64, 128, 256, 256, 512, 512, 512, 512]
        numbers = [2, 2, 2, 1, 2, 1, 2, 1]
        pool_too = [True, True, False, True, False, True, False, True]
        model = Sequential(name="VGG-C")
        input_ = layers.Input(shape=self.input_shape, name="Input")
        model.add(input_)
        if self.with_preprocessing:
            model.add(self.__preprocess())
        for i in range(len(ks)):
            model.add(
                self.__conv_layers(
                    k=ks[i],
                    filters=n_filters[i],
                    numbers=numbers[i],
                    name=f"Conv_{i+1}",
                    maxpool=pool_too[i],
                    layer=i + 1,
                )
            )

        model.add(self.__fc_layer())
        return model

    def _build_D(self):
        ks = [3, 3, 3, 3, 3]
        n_filters = [64, 128, 256, 512, 512]
        numbers = [2, 2, 3, 3, 3]
        pool_too = [True, True, True, True, True]
        model = Sequential(name="VGG-D")
        input_ = layers.Input(shape=self.input_shape, name="Input")
        model.add(input_)
        if self.with_preprocessing:
            model.add(self.__preprocess())
        for i in range(len(ks)):
            model.add(
                self.__conv_layers(
                    k=ks[i],
                    filters=n_filters[i],
                    numbers=numbers[i],
                    name=f"Conv_{i+1}",
                    maxpool=pool_too[i],
                    layer=i + 1,
                )
            )

        model.add(self.__fc_layer())
        return model

    def _build_E(self):
        ks = [3, 3, 3, 3, 3]
        n_filters = [64, 128, 256, 512, 512]
        numbers = [2, 2, 4, 4, 4]
        pool_too = [True, True, True, True, True]
        model = Sequential(name="VGG-E")
        input_ = layers.Input(shape=self.input_shape, name="Input")
        model.add(input_)
        if self.with_preprocessing:
            model.add(self.__preprocess())
        for i in range(len(ks)):
            model.add(
                self.__conv_layers(
                    k=ks[i],
                    filters=n_filters[i],
                    numbers=numbers[i],
                    name=f"Conv_{i+1}",
                    maxpool=pool_too[i],
                    layer=i + 1,
                )
            )

        model.add(self.__fc_layer())
        return model

    def build(self):
        m = self.model.upper()
        ms = ["A", "B", "C", "D", "E"]
        if m not in ms:
            raise ValueError(f"The model {m} does not exist. Try on of {', '.join(ms)}")
        return eval(f"self._build_{m}()")
