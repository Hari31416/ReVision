import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential


def lenet_preprocessing(input_shape=(32, 32, 1), output_shape=(32, 32, 1)):
    """
    Preprocesses the data for LeNet

    Parameters
    ----------
    input_shape : tuple
        The input shape of the data
    output_shape : tuple
        The output shape of the data after preprocessing

    Returns
    -------
    preprocessing : Sequential
        The preprocessing layer
    """
    preprocessing = Sequential(name="preprocessing")
    resize = layers.experimental.preprocessing.Resizing(
        output_shape[0], output_shape[1], name="resize", input_shape=input_shape
    )
    normalization = layers.experimental.preprocessing.Normalization(
        mean=0.0, variance=1.0
    )
    preprocessing.add(resize)
    preprocessing.add(normalization)
    return preprocessing


def built_lenet_og(
    input_shape=(32, 32, 1), with_preprocessing=True, output_shape=(32, 32, 1)
):
    """
    The (almost) original LeNet-5 architecture

    Parameters
    ----------
    input_shape : tuple
        The input shape of the data
    with_preprocessing : bool
        Whether to use the preprocessing layer or not
    output_shape : tuple
        The output shape of the data after preprocessing

    Returns
    -------
    LeNet : Sequential
        The LeNet model
    """
    LeNet = Sequential(name="LeNet_OG")
    if with_preprocessing:
        LeNet.add(
            lenet_preprocessing(input_shape=input_shape, output_shape=output_shape)
        )
    LeNet.add(
        layers.Conv2D(
            6,
            (5, 5),
            activation="tanh",
            padding="valid",
            name="C1",
            input_shape=(32, 32, 1),
        )
    )
    LeNet.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="S2"))
    LeNet.add(layers.Conv2D(16, (5, 5), activation="tanh", padding="valid", name="C3"))
    LeNet.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="S4"))
    LeNet.add(layers.Conv2D(120, (5, 5), activation="tanh", padding="valid", name="C5"))
    LeNet.add(layers.Dense(84, activation="tanh", name="F6"))
    LeNet.add(layers.Dense(10, activation="softmax", name="F7"))
    return LeNet


def built_lenet_mo_1(
    input_shape=(32, 32, 1), with_preprocessing=True, output_shape=(32, 32, 1)
):
    """
    LeNet model with some modifications. See the corresponding notebook/markdown file for more details.

    Parameters
    ----------
    input_shape : tuple
        The input shape of the data
    with_preprocessing : bool
        Whether to use the preprocessing layer or not
    output_shape : tuple
        The output shape of the data after preprocessing

    Returns
    -------
    LeNet : Sequential
        The LeNet model
    """
    LeNet = Sequential(name="LeNet_MO_1")
    if with_preprocessing:
        LeNet.add(lenet_preprocessing(input_shape, output_shape))
    LeNet.add(
        layers.Conv2D(
            6,
            (5, 5),
            activation="relu",
            padding="valid",
            name="C1",
            input_shape=(32, 32, 1),
        )
    )
    LeNet.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="S2"))
    LeNet.add(layers.Conv2D(16, (5, 5), activation="relu", padding="valid", name="C3"))
    LeNet.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="S4"))
    LeNet.add(layers.Conv2D(120, (5, 5), activation="relu", padding="valid", name="C5"))
    LeNet.add(layers.Flatten(name="Flatten"))
    LeNet.add(layers.Dense(84, activation="relu", name="F6"))
    LeNet.add(layers.Dense(10, activation="softmax", name="F7"))
    return LeNet


def built_lenet_mo_2(
    input_shape=(28, 28, 1),
    with_preprocessing=False,
    output_shape=(32, 32, 1),
):
    """
    LeNet model with some more modifications. See the corresponding notebook/markdown file for more details.

    Parameters
    ----------
    input_shape : tuple
        The input shape of the data
    with_preprocessing : bool
        Whether to use the preprocessing layer or not
    output_shape : tuple
        The output shape of the data after preprocessing

    Returns
    -------
    LeNet : Sequential
        The LeNet model
    """
    LeNet = Sequential(name="LeNet_MO_2")
    if with_preprocessing:
        LeNet.add(lenet_preprocessing(input_shape, output_shape))
    LeNet.add(
        layers.Conv2D(
            6,
            (5, 5),
            activation="relu",
            padding="same",
            name="C1",
            input_shape=(28, 28, 1),
        )
    )
    LeNet.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="S2"))
    LeNet.add(layers.Conv2D(16, (5, 5), activation="relu", padding="valid", name="C3"))
    LeNet.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="S4"))
    LeNet.add(layers.Conv2D(120, (5, 5), activation="relu", padding="valid", name="C5"))
    LeNet.add(layers.Flatten(name="Flatten"))
    LeNet.add(layers.Dense(84, activation="relu", name="F6"))
    LeNet.add(layers.Dense(10, activation="softmax", name="F7"))
    return LeNet
