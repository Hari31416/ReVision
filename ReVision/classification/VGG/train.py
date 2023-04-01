import argparse
import os
from ReVision.utilities.data import DataSet
from ReVision.utilities.utils import (
    return_loss,
    return_optimizer,
    return_metric,
    plot_model,
)
from ReVision.classification.VGG.model import VGG

MODELS = "A B C D E".split(" ")


def load_model(args, preprocessing):
    if args.model in MODELS:
        vgg = VGG(
            args.model,
            with_preprocessing=preprocessing,
            input_shape=args.input_shape,
            output_shape=args.output_shape,
        )
        model = vgg.build()
    else:
        raise ValueError("Unknown model")
    return model


def summary_only(args):
    input_shape = args.input_shape
    no_preprocessing = args.no_preprocessing
    preprocessing = not no_preprocessing
    data = None
    if input_shape is None and args.dataset is None:
        raise ValueError("Input shape must be specified. Else specify a dataset")
    if input_shape is not None:
        if input_shape[-1] == 1:
            raise ValueError("Gray scale images are not supported.")
        VGG = load_model(args, preprocessing)
    elif args.dataset is not None:
        dataset = DataSet(args.dataset)
        data = dataset.load()
        VGG = load_model(args, preprocessing)

    VGG.summary(expand_nested=True)
    if args.fig_dir is not None:
        img_dir = args.fig_dir
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        model_name = f"VGG_{args.model}"
        image_file_path = os.path.join(img_dir, model_name + ".png")
        plot_model(VGG, image_file_path)
    return VGG, data


def main(args):
    VGG, dataset = summary_only(args)
    if args.summary_only:
        return
    if args.dataset is None:
        raise ValueError("Dataset must be specified if summary_only is False")
    if dataset is None:
        dataset = DataSet(args.dataset).load()

    (x_train, y_train), (x_test, y_test) = dataset

    optimizer = return_optimizer(args.optimizer, args.lr)
    loss = return_loss(args.loss)
    metrics = [return_metric(metric) for metric in args.metrics]

    VGG.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if x_train.shape[-1] == 1:
        raise ValueError("Gray scale images are not supported.")
    VGG.fit(
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
        default="A",
        help="The model to build",
        choices=MODELS,
    )
    args.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The dataset to use",
        choices=["cifar10", "cifar100"],
    )
    args.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=(224, 224, 3),
        help="The input shape of the model",
    )
    args.add_argument(
        "--output_shape",
        type=int,
        default=1000,
        help="The output shape of the model",
    )
    args.add_argument(
        "--no_preprocessing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args.add_argument(
        "--fig_dir", type=str, help="The directory to save the figures", default=None
    )
    args.add_argument(
        "--summary_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args.add_argument(
        "--batch_size", type=int, default=64, help="The batch size for training"
    )
    args.add_argument(
        "--epochs", type=int, default=10, help="The number of epochs for training"
    )
    args.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="The learning rate for the optimizer",
    )
    args.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="The optimizer to use",
    )
    args.add_argument(
        "--metrics",
        type=list,
        nargs="+",
        default=["accuracy"],
    )
    args.add_argument(
        "--loss",
        type=str,
        default="categorical_crossentropy",
        help="The loss function to use",
    )
    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()
    main(args)
