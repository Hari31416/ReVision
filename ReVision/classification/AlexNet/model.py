import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential


def alexnet_preprocessing(input_shape=(224, 224, 3)):
    """
    Preprocesses the data for LeNet

    Parameters
    ----------
    input_shape : tuple
        The input shape of the data

    Returns
    -------
    preprocessing : Sequential
        The preprocessing layer
    """
    preprocessing = Sequential(name="preprocessing")
    resize = layers.experimental.preprocessing.Resizing(
        224, 224, name="resize", input_shape=input_shape
    )
    normalization = layers.Lambda(
        lambda x: x / 255.0, name="normalization", input_shape=input_shape
    )

    preprocessing.add(resize)
    preprocessing.add(normalization)
    return preprocessing


def built_alexnet(
    with_preprocessing=True, input_shape=(224, 224, 3), output_shape=1000
):
    """
    Builts the AlexNet model

    Parameters
    ----------
    with_preprocessing : bool
        Whether to add preprocessing to the model
    input_shape : tuple
        The input shape of the data
    output_shape : int
        The output shape of the data

    Returns
    -------
    model : Sequential
        The AlexNet model
    """
    model = Sequential(name="AlexNet")
    if with_preprocessing:
        model.add(alexnet_preprocessing(input_shape=input_shape))
    model.add(
        layers.Conv2D(
            96,
            (11, 11),
            strides=(4, 4),
            activation="relu",
            padding="valid",
            name="Conv1",
            input_shape=(224, 224, 3),
        )
    )
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), name="MaxPool1"))
    model.add(
        layers.Conv2D(
            256, (5, 5), strides=(1, 1), activation="relu", padding="same", name="Conv2"
        )
    )
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), name="MaxPool2"))
    model.add(
        layers.Conv2D(
            384, (3, 3), strides=(1, 1), activation="relu", padding="same", name="Conv3"
        )
    )
    model.add(
        layers.Conv2D(
            384, (3, 3), strides=(1, 1), activation="relu", padding="same", name="Conv4"
        )
    )
    model.add(
        layers.Conv2D(
            256, (3, 3), strides=(1, 1), activation="relu", padding="same", name="Conv5"
        )
    )
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), name="MaxPool3"))
    model.add(layers.Flatten(name="Flatten"))
    model.add(layers.Dense(4096, activation="relu", name="FC1"))
    model.add(layers.BatchNormalization(name="BatchNorm1"))
    model.add(layers.Dropout(0.5, name="Dropout1"))
    model.add(layers.Dense(4096, activation="relu", name="FC2"))
    model.add(layers.BatchNormalization(name="BatchNorm2"))
    model.add(layers.Dropout(0.5, name="Dropout2"))
    model.add(layers.Dense(output_shape, activation="softmax", name="FC3"))
    return model
