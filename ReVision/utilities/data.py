import tensorflow as tf
import tensorflow.keras.datasets as tfds
import tensorflow.keras.utils as tfutils


class DataSet:
    """
    Loads a dataset
    """

    def __init__(self, name) -> None:
        """
        Loads a dataset
        """
        self.name = name
        self.str_to_dataset = {
            "mnist": self.load_mnist,
            "cifar10": self.load_cifar10,
            "cifar100": self.load_cifar100,
            "fashion_mnist": self.load_fashion_mnist,
            "image_net": self.load_imagenet,
        }
        self.data = None
        self.input_shape = None
        self.output_shape = None

    def __calc_data_shape(self):
        """
        Calculates the data shape
        """
        (X_train, y_train), (X_test, y_test) = self.data
        self.input_shape = X_train.shape[1:]
        self.output_shape = y_train.shape[1:]

    def load_mnist(self):
        """
        Loads the MNIST digit dataset.
        """
        (X_train, y_train), (X_test, y_test) = tfds.mnist.load_data()
        X_train = X_train.reshape((60000, 28, 28, 1))
        X_test = X_test.reshape((10000, 28, 28, 1))
        y_train = tfutils.to_categorical(y_train)
        y_test = tfutils.to_categorical(y_test)
        return (X_train, y_train), (X_test, y_test)

    def load_cifar10(self):
        """
        Loads the CIFAR10 dataset.
        """
        (X_train, y_train), (X_test, y_test) = tfds.cifar10.load_data()
        y_train = tfutils.to_categorical(y_train)
        y_test = tfutils.to_categorical(y_test)
        return (X_train, y_train), (X_test, y_test)

    def load_cifar100(self):
        """
        Loads the CIFAR100 dataset.
        """
        (X_train, y_train), (X_test, y_test) = tfds.cifar100.load_data()
        y_train = tfutils.to_categorical(y_train)
        y_test = tfutils.to_categorical(y_test)
        return (X_train, y_train), (X_test, y_test)

    def load_fashion_mnist(self):
        """
        Loads the Fashion MNIST dataset.
        """
        (X_train, y_train), (X_test, y_test) = tfds.fashion_mnist.load_data()
        X_train = X_train.reshape((60000, 28, 28, 1))
        X_test = X_test.reshape((10000, 28, 28, 1))
        y_train = tfutils.to_categorical(y_train)
        y_test = tfutils.to_categorical(y_test)
        return (X_train, y_train), (X_test, y_test)

    def load_imagenet(self):
        """
        Loads the ImageNet dataset.
        """
        raise NotImplementedError("ImageNet not implemented")

    def load_data_from_directory(self, directory):
        """
        Loads a dataset from a directory.
        """
        raise NotImplementedError("Loading from directory not implemented")

    def load(self):
        """
        Loads the dataset.
        """
        for key, value in self.str_to_dataset.items():
            if key.lower() == self.name.lower():
                self.data = value()
                self.__calc_data_shape()
                return self.data
        raise ValueError(f"Unknown dataset. Use one of {self.str_to_dataset.keys()}")
