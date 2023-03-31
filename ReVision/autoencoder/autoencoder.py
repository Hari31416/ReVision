import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


class AutoEncoder:
    """AutoEncoder class"""

    def __init__(self, input_shape) -> None:
        """Initialize AutoEncoder class

        Parameters
        ----------
        input_shape : tuple
            Input shape of the data

        Returns
        -------
        None
        """
        self.input_shape = input_shape
        self.encoding_shape = None
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.compiled = False
        self.fitted = False

    def __str__(self) -> str:
        """Print the class

        Returns
        -------
        str
            String representation of the class
        """
        return f"AutoEncoder(input_shape={self.input_shape})"

    def __repr__(self) -> str:
        """Print the class

        Returns
        -------
        str
            String representation of the class
        """
        return self.__str__()

    def encoder_dense(self, inputs, neurons, activation, name="encoder"):
        """Encoder model with dense layers

        Parameters
        ----------
        inputs : tf.keras.layers.Input
            Input layer
        neurons : list
            List of neurons in each layer
        activation : str
            Activation function
        name : str, optional
            Name of the model, by default "encoder"

        Returns
        -------
        (tf.keras.models.Model, tf.keras.layers)
            Encoder model and output layer
        """
        x = inputs
        for i, neuron in enumerate(neurons):
            x = layers.Dense(neuron, activation=activation, name=f"{name}_dense_{i+1}")(
                x
            )
        x = self.bottleneck_dense(x)[1]
        encoder = Model(inputs, x, name=name)
        return encoder, x

    def bottleneck_dense(self, inputs):
        """Bottleneck model with dense layers

        Parameters
        ----------
        inputs : tf.keras.layers
            Input layer

        Returns
        -------
        (tf.keras.models.Model, tf.keras.layers)
            Bottleneck model and output layer
        """
        bottleneck = layers.Dense(
            self.encoding_shape, activation="relu", name="bottleneck"
        )(inputs)
        bottleneck_model = Model(inputs, bottleneck, name="bottleneck_model")
        return bottleneck_model, bottleneck

    def decoder_dense(self, inputs, neurons, activation, name="decoder"):
        """Decoder model with dense layers

        Parameters
        ----------
        inputs : tf.keras.layers
            Input layer
        neurons : list
            List of neurons in each layer
        activation : str
            Activation function
        name : str, optional
            Name of the model, by default "decoder"

        Returns
        -------
        (tf.keras.models.Model, tf.keras.layers)
            Decoder model and output layer
        """
        x = inputs
        for i, neuron in enumerate(neurons):
            x = layers.Dense(neuron, activation=activation, name=f"{name}_dense_{i+1}")(
                x
            )
        outputs = layers.Dense(
            self.input_shape, activation=activation, name=f"{name}_output"
        )(x)
        decoder = Model(inputs, outputs, name=name)
        return decoder, outputs

    def autoencoder_dense(self, neurons, activation):
        """Autoencoder model with dense layers

        Parameters
        ----------
        neurons : list
            List of neurons in each layer
        activation : str
            Activation function

        Returns
        -------
        (tf.keras.models.Model, tf.keras.models.Model, tf.keras.models.Model)
            Encoder, decoder and autoencoder models
        """
        inputs = layers.Input(shape=self.input_shape, name=f"input")
        encoder, encoder_outputs = self.encoder_dense(
            inputs,
            neurons,
            activation,
            name=f"encoder",
        )
        decoder, decoder_outputs = self.decoder_dense(
            encoder_outputs,
            neurons[::-1],
            activation,
            name=f"decoder",
        )
        autoencoder = Model(inputs, decoder_outputs, name="Dense_Autoencoder")
        return encoder, decoder, autoencoder

    def conv_compression_block(self, inputs, filters, kernels, name, activation="relu"):
        """Convolutional compression block. Used for encoding. See the note for more details.

        Parameters
        ----------
        inputs : tf.keras.layers
            Input layer
        filters : int
            Number of filters
        kernels : int
            Kernel size
        name : str
            Name of the block
        activation : str, optional
            Activation function, by default "relu"

        Returns
        -------
        (tf.keras.models.Model, tf.keras.layers)
            Convolutional compression block and output layer
        """
        if not isinstance(filters, int):
            raise ValueError("`filters` must be an integer.")
        if not isinstance(kernels, int):
            raise ValueError("`kernels` must be an integer.")

        x = layers.Conv2D(
            filters,
            kernels,
            padding="same",
            activation=activation,
            name=f"{name}_conv1",
        )(inputs)
        x = layers.Conv2D(
            filters,
            kernels,
            padding="same",
            activation=activation,
            name=f"{name}_conv2",
        )(x)
        x = layers.MaxPool2D(
            2,
            strides=2,
            name=f"{name}_pool",
        )(x)

        block = Model(inputs, x, name=name)
        return block, x

    def conv_expansion_block(self, inputs, filters, kernels, name, activation="relu"):
        """Convolutional expansion block. Used for decoding. See the note for more details.

        Parameters
        ----------
        inputs : tf.keras.layers
            Input layer
        filters : int
            Number of filters
        kernels : int
            Kernel size
        name : str
            Name of the block
        activation : str, optional
            Activation function, by default "relu"

        Returns
        -------
        (tf.keras.models.Model, tf.keras.layers)
            Convolutional expansion block and output layer
        """
        if not isinstance(filters, int):
            raise ValueError("`filters` must be an integer.")
        if not isinstance(kernels, int):
            raise ValueError("`kernels` must be an integer.")

        x = layers.UpSampling2D(
            2,
            name=f"{name}_upsample",
        )(inputs)
        x = layers.Conv2DTranspose(
            filters,
            kernels,
            padding="same",
            activation=activation,
            name=f"{name}_conv1T",
        )(x)
        x = layers.Conv2DTranspose(
            filters,
            kernels,
            padding="same",
            activation=activation,
            name=f"{name}_conv2T",
        )(x)

        block = Model(inputs, x, name=name)
        return block, x

    def bottleneck_conv(self, inputs):
        """Bottleneck model with convolutional layers

        Parameters
        ----------
        inputs : tf.keras.layers
            Input layer

        Returns
        -------
        (tf.keras.models.Model, tf.keras.layers)
            Bottleneck model and output layer
        """
        x = layers.Conv2D(
            1,
            1,
            activation="relu",
            padding="same",
            name="bottleneck_1x1",
        )(inputs)
        bottleneck = Model(inputs, x, name="bottleneck")
        return bottleneck, x

    def encoder_conv(self, inputs, filters, kernels, activation="relu"):
        """Encoder model with convolutional layers

        Parameters
        ----------
        inputs : tf.keras.layers
            Input layer
        filters : list
            List of filters in each block
        kernels : int
            Kernel size
        activation : str, optional
            Activation function, by default "relu"

        Returns
        -------
        (tf.keras.models.Model, tf.keras.layers)
            Encoder model and output layer
        """
        x = inputs
        for i in range(len(filters)):
            _, x = self.conv_compression_block(
                x,
                filters=filters[i],
                kernels=kernels,
                activation=activation,
                name=f"encoder_{i+1}",
            )
        _, x = self.bottleneck_conv(x)
        encoder = Model(inputs, x, name="encoder")
        return encoder, x

    def decoder_conv(self, inputs, filters, kernels, activation="relu"):
        """Decoder model with convolutional layers

        Parameters
        ----------
        inputs : tf.keras.layers
            Input layer
        filters : list
            List of filters in each block
        kernels : int
            Kernel size
        activation : str, optional
            Activation function, by default "relu"

        Returns
        -------
        (tf.keras.models.Model, tf.keras.layers)
            Decoder model and output layer
        """
        x = inputs
        for i in range(len(filters)):
            _, x = self.conv_expansion_block(
                x,
                filters=filters[i],
                kernels=kernels,
                activation=activation,
                name=f"decoder_{i+1}",
            )

        output_channels = self.input_shape[-1]
        x = layers.Conv2D(
            output_channels,
            1,
            activation="relu",
            padding="same",
            name=f"output",
        )(x)
        decoder = Model(inputs, x, name="decoder")
        return decoder, x

    def autoencoder_conv(self, filters, kernels, activation):
        """Autoencoder model with convolutional layers

        Parameters
        ----------
        inputs : tf.keras.layers
            Input layer
        filters : list
            List of filters in each block
        kernels : int
            Kernel size
        activation : str, optional
            Activation function, by default "relu"

        Returns
        -------
        (tf.keras.models.Model, tf.keras.models.Model, tf.keras.models.Model)
            Encoder, decoder and autoencoder models
        """
        if not isinstance(filters, list):
            raise ValueError("`filters` must be a list of integers")

        if not isinstance(kernels, int):
            raise ValueError("`kernels` must be an integer.")
        inputs = layers.Input(shape=self.input_shape, name="input")
        encoder, encoder_outputs = self.encoder_conv(
            inputs,
            filters,
            kernels,
            activation,
        )
        decoder, decoder_outputs = self.decoder_conv(
            encoder_outputs,
            filters[::-1],
            kernels,
            activation,
        )
        if inputs.shape != decoder_outputs.shape:
            raise ValueError(
                "Input and output shapes must be the same. Please check your model architecture. (Maxpooling from odd number of pixels be the reason.)"
            )
        autoencoder = Model(inputs, decoder_outputs, name="Conv_Autoencoder")

        return encoder, decoder, autoencoder

    def build(
        self,
        neurons=None,
        encoding_shape=None,
        filters=None,
        kernels=3,
        activation="relu",
    ):
        """Builds the autoencoder model given parameters

        Parameters
        ----------
        neurons : list, optional
            List of neurons in each layer. Must be provided for dense autoencoder, by default None (Not required for conv autoencoder)
        encoding_shape : int, optional
            Number of neurons in the bottleneck layer. Must be provided for dense autoencoder, by default None (Not required for conv autoencoder)
        filters : list, optional
            List of filters in each block. Must be provided for conv autoencoder, by default None (Not required for dense autoencoder)
        kernels : int, optional
            Kernel size, by default 3 (Only required for conv autoencoder)
        activation : str, optional
            Activation function, by default "relu"

        Raises
        ------
        ValueError
            If `neurons` is not provided for dense autoencoder
        ValueError
            If `encoding_shape` is not provided for dense autoencoder
        ValueError
            If `filters` is not provided for conv autoencoder
        ValueError
            If `filters` is not provided as a list for conv autoencoder

        Returns
        -------
        None
        """
        if filters is not None:
            if not isinstance(filters, list):
                raise ValueError(
                    "For conv autoencoder, you must provide the `filters` as a list"
                )
            (self.encoder, self.decoder, self.autoencoder) = self.autoencoder_conv(
                filters,
                kernels,
                activation,
            )
            return
        elif neurons is not None:
            if encoding_shape is None:
                raise ValueError(
                    "For dense autoencoder, you must provide the `encoding_shape`."
                )
            else:
                self.encoding_shape = encoding_shape
            (self.encoder, self.decoder, self.autoencoder) = self.autoencoder_dense(
                neurons,
                activation,
            )
            return

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        """Compile autoencoder model

        Parameters
        ----------
        optimizer : tf.keras.optimizers or str
            Optimizer for the model
        loss : tf.keras.losses or str
            Loss function for the model
        metrics : list, optional
            List of metrics to be evaluated by the model during training and testing, by default None
        **kwargs
            Keyword arguments for `tf.keras.Model.compile()`

        Raises
        ------
        ValueError
            If autoencoder model is not built

        Returns
        -------
        None
        """
        if self.autoencoder is None:
            raise ValueError("Autoencoder model is not built. Call `build()` first.")
        self.autoencoder.compile(
            optimizer=optimizer, loss=loss, metrics=metrics, **kwargs
        )
        self.compiled = True

    def summary(self):
        """Print model summary

        Raises
        ------
        ValueError
            If autoencoder model is not built

        Returns
        -------
        None
        """
        if self.autoencoder is None:
            raise ValueError("Autoencoder model is not built. Call `build()` first.")
        self.autoencoder.summary()

    def fit(self, X, epochs, validation_data=None, **kwargs):
        """Fit autoencoder model

        Parameters
        ----------
        X : (tensorflow.python.data.ops.dataset_ops.BatchDataset, tensorflow.python.data.ops.dataset_ops.BatchDataset)
            Training data (Assuming X is a tuple of (X, X))
        epochs : int
            Number of epochs
        validation_data : (tensorflow.python.data.ops.dataset_ops.BatchDataset, tensorflow.python.data.ops.dataset_ops.BatchDataset), optional
            Validation data (Assuming X is a tuple of (X, X)), by default None
        **kwargs
            Keyword arguments for `tf.keras.Model.fit()`

        Raises
        ------
        ValueError
            If autoencoder model is not built
        ValueError
            If autoencoder model is not compiled

        Returns
        -------
        tf.keras.callbacks.History
            History object
        """
        if self.autoencoder is None:
            raise ValueError("Autoencoder model is not built. Call `build()` first.")
        if not self.compiled:
            raise ValueError(
                "Autoencoder model is not compiled. Call `compile()` first."
            )
        steps_per_epoch = len(X)
        if validation_data is not None:
            validation_steps = len(validation_data)
        else:
            validation_steps = None
        history = self.autoencoder.fit(
            X,
            epochs=epochs,
            validation_data=validation_data,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            **kwargs,
        )
        self.fitted = True
        return history

    def encode(self, inputs):
        """Encode inputs

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor

        Raises
        ValueError
            If autoencoder model is not built

        Returns
        ------
        tf.Tensor
            Encoded tensor
        """
        if self.encoder is None:
            raise ValueError("Autoencoder model is not built. Call `build()` first.")
        return self.encoder(inputs)

    def decode(self, inputs):
        """Decode inputs

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor

        Raises
        ValueError
            If autoencoder model is not built

        Returns
        ------
        tf.Tensor
            Decoded tensor
        """
        if self.decoder is None:
            raise ValueError("Autoencoder model is not built. Call `build()` first.")
        return self.decoder(inputs)

    def evaluate(self, images, original_shape, encoded_shape, num_images):
        """Evaluates and plots the encoding learned by autoencoder model

        Parameters
        ----------
        images : tf.Tensor
            Images to be evaluated
        original_shape : tuple
            Shape of the original image
        encoded_shape : tuple
            Shape of the encoded image
        num_images : int
            Number of images to be evaluated

        Raises
        ------
        ValueError
            If the model is not fitted

        Returns
        -------
        (plt.figure, plt.axes)
            Figure and axes of the plot
        """
        if not self.fitted:
            raise ValueError("The model must be trained first.")
        if num_images > len(images):
            print(
                f"Number of samples provided ({len(num_images)}) is less than the number of images specified to plot ({num_images})."
            )
        encoded_images = self.encode(images)
        decoded_images = self.decode(encoded_images)

        def plot_images(axes, encoded_image, decoded_image, original_image):
            """Plot images"""
            for i, image in enumerate([original_image, encoded_image, decoded_image]):
                axes[i].imshow(image)
                axes[i].set_axis_off()
            return axes

        fig, axes = plt.subplots(3, num_images, figsize=(20, 8))

        for i in range(num_images):
            plot_images(
                axes[:, i],
                encoded_images[i].numpy().reshape(encoded_shape),
                decoded_images[i].numpy().reshape(original_shape),
                images[i].numpy().reshape(original_shape),
            )
        annotations = ["Original", "Encoded", "Decoded"]
        for i, annotation in enumerate(annotations):
            axes[i, num_images // 2].set_title(
                annotation,
                fontsize=25,
                fontweight="bold",
                fontfamily="serif",
                color="red",
            )
        plt.tight_layout()
        fig.show()
        return fig, axes
