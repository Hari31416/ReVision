import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from IPython.display import display as display_fn
from imageio import mimwrite


class StyleTransfer:
    """A class for neural style transfer. Uses the Inception model to extract features from the content and style images."""

    content_layers = ["conv2d_88", "conv2d_91", "conv2d_92", "conv2d_85", "conv2d_93"]
    style_layers = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4"]
    content_and_style_layers = content_layers + style_layers

    NUM_CONTENT_LAYERS = len(content_layers)
    NUM_STYLE_LAYERS = len(style_layers)

    def __init__(self, content_image_path, style_image_path) -> None:
        """Initializes the class

        Parameters
        ----------
        content_image_path : str
            path to the content image
        style_image_path : str
            path to the style image

        Returns
        -------
        None
        """
        self.content_image_path = content_image_path
        self.style_image_path = style_image_path
        self.model = None

    def tensor_to_image(self, tensor):
        """converts a tensor to an image"""
        tensor_shape = tf.shape(tensor)
        number_elem_shape = tf.shape(tensor_shape)
        if number_elem_shape > 3:
            assert tensor_shape[0] == 1, "There are more than one image"
            tensor = tensor[0]
        return tf.keras.preprocessing.image.array_to_img(tensor)

    def load_image(self, path_to_img):
        """loads an image as a tensor and scales it to 512 pixels"""
        max_dim = 512
        image = tf.io.read_file(path_to_img)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        shape = tf.shape(image)[:-1]
        shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        image = tf.image.resize(image, new_shape)
        image = image[tf.newaxis, :]
        image = tf.image.convert_image_dtype(image, tf.uint8)

        return image

    def imshow(self, image, title=""):
        """displays an image"""
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        plt.title(title)

    def show_images_with_style(self, images, titles=[]):
        """displays a row of images with corresponding titles"""
        if len(images) != len(titles):
            return

        plt.figure(figsize=(20, 12))
        for idx, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(1, len(images), idx + 1)
            plt.xticks([])
            plt.yticks([])
            self.imshow(image, title)
        plt.show()

    def preprocess_image(self, image):
        """preprocesses a given image to use with Inception model"""
        image = tf.cast(image, dtype=tf.float32)
        image = (image / 127.5) - 1.0

        return image

    def display_images(self):
        """displays the content and style images"""
        content_image = self.load_image(self.content_image_path)
        style_image = self.load_image(self.style_image_path)

        self.show_images_with_style(
            [content_image, style_image],
            titles=[f"Content image", f"Style image"],
        )

    def gram_matrix(self, input_tensor):
        """Calculates the gram matrix and divides by the number of locations

        Parameters
        ----------
        input_tensor : tensor
            tensor to calculate the gram matrix from

        Returns
        -------
        tensor
            gram matrix of the input tensor
        """

        gram = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)

        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        scaled_gram = gram / num_locations

        return scaled_gram

    def get_features(self, image, type=None):
        """Returns the features of the image

        Parameters
        ----------
        image : tensor
            image to extract features from
        type : str
            type of features to extract. Either "style" or "content". If `None` is provided, both
            content and style features are returned

        Returns
        -------
        list
            list of features of the content and style images
        """
        preprocessed_image = self.preprocess_image(image)
        outputs = self.model(preprocessed_image)

        if type == "style":
            style_outputs = outputs[self.NUM_CONTENT_LAYERS :]
            gram_style_features = [
                self.gram_matrix(style_output) for style_output in style_outputs
            ]
            return gram_style_features

        elif type == "content":
            content_outputs = outputs[: self.NUM_CONTENT_LAYERS]
            return content_outputs

        else:
            style_outputs = outputs[self.NUM_CONTENT_LAYERS :]
            content_outputs = outputs[: self.NUM_CONTENT_LAYERS]
            gram_style_features = [
                self.gram_matrix(style_output) for style_output in style_outputs
            ]
            return content_outputs + gram_style_features

    def _loss(self, features, targets, type="style"):
        """Returns the loss of fearure and target. This is just the mean square error.

        features : list
            list of features of the content and style images
        target : list
            list of features of the content and style images
        type : str
            type of loss to calculate. Either "style" or "content"
        """
        loss = tf.reduce_mean(tf.square(features - targets))
        if type == "content":
            loss = loss * 0.5
        return loss

    def get_loss(self, features, target, alpha, beta):
        """Returns the total loss of the style and content images

        Parameters
        ----------
        features : list
            list of features of the content and style images
        target : list
            list of features of the content and style images
        alpha : float
            weight of the content loss
        beta : float
            weight of the style loss

        Returns
        -------
        loss : float
            total loss of the style and content images
        """
        style_features = features[self.NUM_CONTENT_LAYERS :]
        content_features = features[: self.NUM_CONTENT_LAYERS]
        style_targets = target[self.NUM_CONTENT_LAYERS :]
        content_targets = target[: self.NUM_CONTENT_LAYERS]
        style_loss = 0
        content_loss = 0

        for i in range(self.NUM_STYLE_LAYERS):
            style_loss += self._loss(style_features[i], style_targets[i], type="style")
        for i in range(self.NUM_CONTENT_LAYERS):
            content_loss += self._loss(
                content_features[i], content_targets[i], type="content"
            )

        style_loss = beta * style_loss / self.NUM_STYLE_LAYERS
        content_loss = alpha * content_loss / self.NUM_CONTENT_LAYERS
        loss = content_loss + style_loss
        return loss

    def calculate_gradients(self, image, target, alpha, beta):
        """Calculates the gradients of the loss with respect to the image"""
        with tf.GradientTape() as tape:
            features = self.get_features(image, "all")
            loss = self.get_loss(features, target, alpha, beta)
        gradients = tape.gradient(loss, image)
        return gradients, loss

    def update_image(self, image, target, alpha, beta, optimizer):
        """Updates the image by calculating the gradients and applying them to the image"""
        gradients, loss = self.calculate_gradients(image, target, alpha, beta)
        optimizer.apply_gradients([(gradients, image)])
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0))
        return loss

    def load_model(self):
        """Creates a inception model that returns a list of intermediate output values"""
        K.clear_session()
        inception = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet"
        )
        inception.trainable = False
        output_layers = self.content_and_style_layers

        model = tf.keras.models.Model(
            [inception.input],
            [inception.get_layer(name).output for name in output_layers],
        )
        self.model = model
        return model

    def stylize_image(
        self,
        alpha=1,
        beta=0.1,
        epochs=10,
        steps_per_epoch=10,
        show_images=True,
        image_frequency=2,
        notebook=False,
        lr=None,
    ):
        """Stylizes the image using the style and content images

        Parameters
        ----------
        alpha : float, optional
            Content weight, by default 1
        beta : float, optional
            Style weight, by default 0.1
        epochs : int, optional
            Number of epochs, by default 10
        steps_per_epoch : int, optional
            Number of steps per epoch, by default 10
        show_images : bool, optional
            Show images, by default True
        image_frequency : int, optional
            Frequency of images to show, by default 2
        notebook : bool, optional
            If the code is running on a notebook, by default False
        lr : float, optional
            Learning rate, by default None

        Returns
        -------
        [PIL.Image]
            List of images
        """
        if self.model is None:
            K.clear_session()
            _ = self.load_model()
        style_image = self.load_image(self.style_image_path)
        content_image = self.load_image(self.content_image_path)

        style_target = self.get_features(style_image, "style")
        content_target = self.get_features(content_image, "content")

        target = content_target + style_target
        image = tf.cast(content_image, dtype=tf.float32)
        image = tf.Variable(image)
        images = []
        if lr is None:
            lr = 40.0
        optimizer = tf.optimizers.Adam(
            tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr, decay_steps=100, decay_rate=0.80
            )
        )
        img = None
        for epoch in range(epochs):
            for step in range(steps_per_epoch):

                loss = self.update_image(image, target, alpha, beta, optimizer)
                display_image = self.tensor_to_image(image)
                images.append(display_image)
                if show_images:
                    if (step) % image_frequency == 0:
                        if notebook:
                            display_image = self.tensor_to_image(image)
                            display_fn(
                                display_image,
                                clear=True,
                            )
                        else:
                            im = np.array(display_image)
                            if img is None:
                                img = plt.imshow(im)
                            else:
                                img.set_data(im)
                            plt.pause(0.1)
                            plt.draw()

                print(f"Epoch: {epoch+1} | Step {step+1} | Loss {loss}", end="\r")
        return images

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
