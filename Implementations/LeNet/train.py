import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
import argparse
import ast
import os
from model import built_lenet_og, built_lenet_mo_1, built_lenet_mo_2


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    print(x_train.max())
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)


def main(args):

    if args.model == "LeNet_OG":
        LeNet = built_lenet_og(
            with_preprocessing=args.preprocessing,
            input_shape=args.input_shape,
            output_shape=args.output_shape,
        )
    elif args.model == "LeNet_MO_1":
        LeNet = built_lenet_mo_1(
            with_preprocessing=args.preprocessing,
            input_shape=args.input_shape,
            output_shape=args.output_shape,
        )
    elif args.model == "LeNet_MO_2":
        LeNet = built_lenet_mo_2(
            with_preprocessing=args.preprocessing,
            input_shape=args.input_shape,
            output_shape=args.output_shape,
        )
    else:
        raise ValueError("Unknown model")

    LeNet.summary()

    if args.fig_dir is not None:
        img_dir = args.fig_dir
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        image_file_path = os.path.join(img_dir, args.model + ".png")
        tf.keras.utils.plot_model(
            LeNet,
            to_file=image_file_path,
            show_shapes=True,
            show_layer_names=True,
        )

    if args.summary_only:
        return

    (x_train, y_train), (x_test, y_test) = load_data()
    LeNet.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    LeNet.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
    )


def arg_parse():
    args = argparse.ArgumentParser(add_help=True)
    args.add_argument(
        "--model",
        type=str,
        default="LeNet_OG",
        help="The model to build",
        choices=["LeNet_OG", "LeNet_MO_1", "LeNet_MO_2"],
    )
    args.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=(32, 32, 1),
        type=int,
        help="The input shape of the model",
    )
    args.add_argument(
        "--output_shape",
        type=int,
        nargs="+",
        default=(32, 32, 1),
        type=int,
        help="The output shape of the preprocessing layer",
    )
    args.add_argument(
        "--preprocessing",
        type=ast.literal_eval,
        default=False,
        help="Whether to use the preprocessing layer or not",
    )
    args.add_argument(
        "--fig_dir", type=str, help="The directory to save the figures", default=None
    )
    args.add_argument(
        "--summary_only",
        type=ast.literal_eval,
        default=False,
        help="Whether to only print summary",
    )
    args.add_argument(
        "--batch_size", type=int, default=128, help="The batch size for training"
    )
    args.add_argument(
        "--epochs", type=int, default=10, help="The number of epochs for training"
    )
    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()
    main(args=args)
