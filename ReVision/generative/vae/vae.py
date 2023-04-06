import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from IPython.display import display as display_fn
import numpy as np
from PIL import Image
from imageio import mimwrite


class Sampling(tf.keras.layers.Layer):
    """Uses (mu, sigma) to sample z, the latent vector."""

    def call(self, inputs):
        """Call method

        Parameters
        ----------
        inputs : tuple
            Tuple of (mu, sigma)

        Returns
        -------
        tf.Tensor
            Sampled latent vector
        """
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return mu + tf.exp(0.5 * sigma) * epsilon


class VAE:
    """VAE class"""

    def __init__(self, input_shape) -> None:
        """Initialize VAE class

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
        self.vae = None
        self.compiled = False
        self.fitted = False

    def __str__(self) -> str:
        """Print the class

        Returns
        -------
        str
            String representation of the class
        """
        return f"VAE(input_shape={self.input_shape})"

    def __repr__(self) -> str:
        """Print the class

        Returns
        -------
        str
            String representation of the class
        """
        return self.__str__()

    def compression_block(self, inputs, filters, kernels, name, activation="relu"):
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
            name=f"{name}1",
        )(inputs)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Conv2D(
            filters,
            kernels,
            padding="same",
            strides=2,
            activation=activation,
            name=f"{name}2",
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)

        block = Model(inputs, x, name=name)
        return block, x

    def expansion_block(self, inputs, filters, kernels, name, activation="relu"):
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
        x = layers.Conv2DTranspose(
            filters,
            kernels,
            padding="same",
            strides=2,
            activation=activation,
            name=f"{name}1T",
        )(inputs)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.Conv2DTranspose(
            filters,
            kernels,
            padding="same",
            activation=activation,
            name=f"{name}2T",
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn3")(x)

        block = Model(inputs, x, name=name)
        return block, x

    def bottleneck(self, inputs):
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
        x = layers.Flatten()(inputs)
        x = layers.Dense(self.bottleneck_shape, activation="relu", name="dense")(x)
        x = layers.BatchNormalization(name="bottleneck_bn")(x)
        mu = layers.Dense(self.encoding_shape, name="mu")(x)
        sigma = layers.Dense(self.encoding_shape, name="sigma")(x)
        return mu, sigma

    def encoder_(self, inputs, filters, kernels, activation="relu"):
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
            _, x = self.compression_block(
                x,
                filters=filters[i],
                kernels=kernels,
                activation=activation,
                name=f"encoder_{i+1}",
            )
        batch_shape = x.shape
        self.batch_shape = batch_shape
        mu, sigma = self.bottleneck(x)
        z = Sampling()([mu, sigma])
        encoder = Model(inputs, [mu, sigma, z], name="encoder")
        return encoder, z, mu, sigma

    def decoder_(self, inputs, filters, kernels, activation="relu"):
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
        units = self.batch_shape[1] * self.batch_shape[2] * self.batch_shape[3]
        x = layers.Dense(units, activation="relu", name="decoder_dense")(x)
        x = tf.keras.layers.BatchNormalization(name="decoder_bn")(x)
        x = tf.keras.layers.Reshape(
            (self.batch_shape[1], self.batch_shape[2], self.batch_shape[3]),
            name="decoder_reshape",
        )(x)

        for i in range(len(filters)):
            _, x = self.expansion_block(
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
            activation="sigmoid",
            padding="same",
            name=f"output",
        )(x)
        decoder = Model(inputs, x, name="decoder")
        return decoder, x

    def vae_(self, filters, kernels, activation):
        """VAE model with convolutional layers

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
            Encoder, decoder and vae models
        """
        if not isinstance(filters, list):
            raise ValueError("`filters` must be a list of integers")

        if not isinstance(kernels, int):
            raise ValueError("`kernels` must be an integer.")
        self.inputs = layers.Input(shape=self.input_shape, name="input")
        encoder, z, sigma, mu = self.encoder_(
            self.inputs,
            filters,
            kernels,
            activation,
        )
        decoder, decoder_outputs = self.decoder_(
            z,
            filters[::-1],
            kernels,
            activation,
        )
        if self.inputs.shape != decoder_outputs.shape:
            raise ValueError(
                "Input and output shapes must be the same. Please check your model architecture. (Maxpooling from odd number of pixels be the reason.)"
            )
        vae = Model(self.inputs, decoder_outputs, name="VAE")
        kl_loss = self.kl_loss(mu, sigma)
        vae.add_loss(kl_loss)
        return encoder, decoder, vae

    def kl_loss(self, mu, sigma):
        """KL divergence loss

        Parameters
        ----------
        mu : tf.Tensor
            Mean
        sigma : tf.Tensor
            Standard deviation

        Returns
        -------
        tf.Tensor
            KL divergence loss
        """
        kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        return kl_loss

    def build(
        self,
        bottleneck_shape=100,
        encoding_shape=256,
        filters=None,
        kernels=3,
        activation="relu",
    ):
        """Build the VAE model

        Parameters
        ----------
        bottleneck_shape : int, optional
            Bottleneck shape, by default 100
        encoding_shape : int, optional
            Encoding (latent) shape, by default 256
        filters : list, optional
            List of filters in each block, by default None which uses [32, 64]
        kernels : int, optional
            Kernel size, by default 3
        activation : str, optional
            Activation function, by default "relu"

        Returns
        -------
        None
        """
        self.bottleneck_shape = bottleneck_shape
        self.encoding_shape = encoding_shape
        if filters is None:
            print("Warning: Using default filters")
            filters = [32, 64]
        self.filters = filters
        self.kernels = kernels
        self.activation = activation
        self.encoder, self.decoder, self.vae = self.vae_(
            self.filters, self.kernels, self.activation
        )

    def summary(self):
        """Print model summary

        Raises
        ------
        ValueError
            If vae model is not built

        Returns
        -------
        None
        """
        if self.vae is None:
            raise ValueError("VAE model is not built. Call `build()` first.")
        self.vae.summary()

    def train(
        self,
        X,
        epochs,
        optimizer="adam",
        lr=0.001,
        notebook=False,
        show_images=True,
        image_frequency=10,
    ):
        """Train the model

        Parameters
        ----------
        X : tf.data.Dataset
            Dataset (Must be a tf.data.Dataset)
        epochs : int
            Number of epochs
        optimizer : str, optional
            Optimizer, by default "adam"
        lr : float, optional
            Learning rate, by default 0.001
        notebook : bool, optional
            Set to True if you are using a notebook, by default False
        show_images : bool, optional
            Set to True to show images, by default True
        image_frequency : int, optional
            Frequency at which images are shown, by default 10

        Raises
        ------
        ValueError
            If vae model is not built

        Returns
        -------
        images : list
            List of images
        """

        if self.vae is None:
            raise ValueError("VAE model is not built. Call `build()` first.")
        optimizer = tf.keras.optimizers.get(optimizer)
        optimizer.learning_rate = lr

        input_pixels = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]

        loss_metric = tf.keras.metrics.Mean()
        mse_loss = tf.keras.losses.MeanSquaredError()
        bce_loss = tf.keras.losses.BinaryCrossentropy()
        random_vector_for_generation = tf.random.normal(shape=[16, self.encoding_shape])
        fig, axs = plt.subplots(4, 4, figsize=(16, 16))
        images = []
        img = None
        batches = len(X)
        for epoch in range(epochs):
            for batch, X_batch in enumerate(X):
                with tf.GradientTape() as tape:
                    reconstructed = self.vae(X_batch)

                    flatten_inputs = tf.reshape(X_batch, [-1, input_pixels])
                    flatten_outputs = tf.reshape(reconstructed, [-1, input_pixels])
                    loss = bce_loss(flatten_inputs, flatten_outputs)
                    loss = loss * input_pixels
                    loss += sum(self.vae.losses)

                grads = tape.gradient(loss, self.vae.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.vae.trainable_weights))

                loss_metric(loss)
                loss_value = loss_metric.result()
                display_image = self.show_image_progress(
                    axs,
                    fig,
                    epoch,
                    batch,
                    random_vector_for_generation,
                    loss_value,
                )
                images.append(display_image)
                if show_images:
                    if batch % image_frequency == 0:
                        if notebook:
                            display_fn(display_image, clear=True)
                        else:
                            im = np.array(display_image)
                            if img is None:
                                img = plt.imshow(im)
                            else:
                                img.set_data(im)
                            plt.pause(0.001)
                            plt.draw()

                print(
                    f"Epoch: {epoch+1}/{epochs} | Batch {batch+1}/{batches} | Mean Loss {loss_value:.2f}",
                    end="\r",
                )
                loss_metric.reset_states()
        return images

    def encode(self, inputs):
        """Encode inputs

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor

        Raises
        ValueError
            If vae model is not built

        Returns
        ------
        tf.Tensor
            Encoded tensor
        """
        if self.encoder is None:
            raise ValueError("VAE model is not built. Call `build()` first.")
        return self.encoder(inputs)

    def decode(self, inputs):
        """Decode inputs

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor

        Raises
        ValueError
            If vae model is not built

        Returns
        ------
        tf.Tensor
            Decoded tensor
        """
        if self.decoder is None:
            raise ValueError("VAE model is not built. Call `build()` first.")
        return self.decoder(inputs)

    def plt2arr(self, fig, draw=True):
        """
        converts a matplotlib figure to a numpy array
        """
        if draw:
            fig.canvas.draw()
        rgba_buf = fig.canvas.buffer_rgba()
        (w, h) = fig.canvas.get_width_height()
        rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
        return rgba_arr

    def show_image_progress(self, axs, fig, epoch, batch, samples, loss=0):
        """Show image progress

        Parameters
        ----------
        axs : matplotlib.axes.Axes
            Axes
        fig : matplotlib.figure.Figure
            Figure
        epoch : int
            Epoch number
        batch : int
            batch number
        samples : tf.Tensor
            Samples
        loss : float, optional
            Loss, by default 0
        Returns
        -------
        image : PIL.Image
            Image
        """
        preds = self.decode(samples)
        preds = preds.numpy()
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            img = preds[i] * 255
            img = img.astype("int32")
            ax.imshow(img)
            ax.axis("off")
        fig.suptitle(f"Epoch {epoch + 1}, batch {batch}, Mean loss={loss:.2f}")
        array = self.plt2arr(fig)
        image = Image.fromarray(array)
        image = image.resize((1024, 1024))
        return image

    def make_animation(self, images, name, duration):
        """Creates an animation from a list of images and saves it to the given name

        Parameters
        ----------
        images : list
            List of images to be animated
        name : str
            Name of the file to be saved
        duration : float
            Duration of the animation in seconds

        Raises
        ------
        ValueError
            If the extension is not gif or mp4

        Returns
        -------
        None
        """
        extension = name.split(".")[-1]
        if extension == "gif":
            mimwrite(name, images, duration=duration)
        elif extension == "mp4":
            fps = len(images) / duration
            fps = max(int(fps), 1)
            mimwrite(name, images, fps=fps)
        else:
            raise ValueError("Please enter a valid extension")
