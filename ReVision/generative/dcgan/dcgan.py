import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from IPython.display import display as display_fn
import numpy as np
from PIL import Image
from imageio import mimwrite


class DCGAN:
    """A class for deep convolutional generative adversarial networks."""

    def __init__(self, latent_dim=32, image_shape=(28, 28, 1)):
        """Initialize the DCGAN class.

        Parameters
        ----------
        latent_dim : int
            The dimensionality of the latent space.
        image_shape : tuple
            The shape of the images to be generated.
        """

        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.__setup()

    def __setup(self):
        """Does set up"""
        if self.image_shape[0] % 4 != 0 or self.image_shape[1] % 4 != 0:
            raise ValueError("Image shape must be divisible by 4")

        self.generator = self.generator_()

        self.discriminator = self.discriminator_()

        self.dcgan = Sequential([self.generator, self.discriminator], name="DCGAN")

    def generator_(self):
        """Create a generator model.

        Returns
        -------
        g : Model
            The generator model.
        """
        a = self.image_shape[0] // 4
        b = self.image_shape[1] // 4
        c = self.image_shape[2]
        g = Sequential(
            [
                layers.Dense(a * b * 128, input_shape=[self.latent_dim]),
                layers.Reshape([a, b, 128]),
                layers.BatchNormalization(),
                layers.Conv2DTranspose(
                    64,
                    kernel_size=5,
                    strides=2,
                    padding="SAME",
                    activation="selu",
                ),
                layers.BatchNormalization(),
                layers.Conv2DTranspose(
                    c,
                    kernel_size=5,
                    strides=2,
                    padding="SAME",
                    activation="tanh",
                ),
            ],
            name="Generator",
        )
        return g

    def discriminator_(self):
        """Create a discriminator model.

        Returns
        -------
        d : Model
            The discriminator model.
        """

        d = Sequential(
            [
                layers.Conv2D(
                    64,
                    kernel_size=5,
                    strides=2,
                    padding="SAME",
                    activation=layers.LeakyReLU(0.2),
                    input_shape=self.image_shape,
                ),
                layers.Dropout(0.4),
                layers.Conv2D(
                    128,
                    kernel_size=5,
                    strides=2,
                    padding="SAME",
                    activation=layers.LeakyReLU(0.2),
                ),
                layers.Dropout(0.4),
                layers.Flatten(),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="Discriminator",
        )
        return d

    def compile(self, loss="binary_crossentropy", optimizer="adam", **kwargs):
        """Compile the DCGAN model.

        Parameters
        ----------
        loss : str
            The loss function to use.
        optimizer : str
            The optimizer to use.
        """

        self.discriminator.compile(loss=loss, optimizer=optimizer, **kwargs)
        self.discriminator.trainable = False
        self.dcgan.compile(loss=loss, optimizer=optimizer, **kwargs)

    def train(
        self,
        X,
        epochs,
        show_image_frequency=20,
        save_image_frequency=10,
        notebook=False,
    ):
        """Train the DCGAN model.

        Parameters
        ----------
        X : array-like
            The training data.
        epochs : int
            The number of epochs to train for.
        show_image_frequency : int
            The number of batches to train before visualizing the results.
        save_image_frequency : int
            The number of batches to train before saving the results.
        notebook : bool
            Whether or not the code is being run in a notebook.

        Returns
        -------
        images : list
            A list of images generated during training.
        """
        batches = len(X)
        fig, axs = plt.subplots(6, 6, figsize=(9, 9))
        img = None
        images = []
        noise_for_generation = tf.random.normal(shape=[36, self.latent_dim])

        for epoch in range(epochs):
            batch = 0
            for X_batch in X:
                batch_size = X_batch.shape[0]
                noise = tf.random.normal(shape=[batch_size, self.latent_dim])

                fake_images = self.generator(noise)

                mixed_images = tf.concat([fake_images, X_batch], axis=0)

                discriminator_labels = tf.constant(
                    [[0.0]] * batch_size + [[1.0]] * batch_size
                )

                self.discriminator.trainable = True

                self.discriminator.train_on_batch(mixed_images, discriminator_labels)

                noise = tf.random.normal([batch_size, self.latent_dim])

                generator_labels = tf.constant([[1.0]] * batch_size)

                self.discriminator.trainable = False

                self.dcgan.train_on_batch(noise, generator_labels)
                batch += 1

                print(f"Epoch {epoch+1}/{epochs} Batch {batch}/ {batches}", end="\r")

                smaller_freq = min(show_image_frequency, save_image_frequency)
                if smaller_freq and batch % smaller_freq == 0:
                    fake_images = self.generator(noise_for_generation)
                    display_image = self.show_image_progress(
                        axs,
                        fig,
                        epoch,
                        batch,
                        fake_images,
                    )

                if save_image_frequency and batch % save_image_frequency == 0:
                    images.append(display_image)

                if show_image_frequency and batch % show_image_frequency == 0:
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
        if images:
            return images

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

    def show_image_progress(self, axs, fig, epoch, batch, images):
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
        Returns
        -------
        image : PIL.Image
            Image
        """
        axs = axs.flatten()
        if images.shape[-1] == 1:
            images = np.squeeze(images, axis=-1)
            cmap = "binary"
        else:
            cmap = None
        for i, ax in enumerate(axs):
            img = images[i]
            ax.imshow(img, cmap=cmap)
            ax.axis("off")
        fig.suptitle(f"Epoch {epoch + 1}, batch {batch}")
        array = self.plt2arr(fig)
        image = Image.fromarray(array)
        image = image.resize((1008, 1008))
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
