import argparse
import os
from ReVision.utilities.data import DataSet
from ReVision.utilities.utils import (
    return_loss,
    return_optimizer,
    return_metric,
    plot_model,
)
from ReVision.MobileNet.model import MobileNet, RHO_TO_RES


def load_model(args, preprocessing):
    mb = MobileNet(
        with_preprocessing=preprocessing,
        input_shape=args.input_shape,
        output_shape=args.output_shape,
    )
    model = mb.build(name=args.model, alpha=args.alpha, rho=args.rho)
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
        model = load_model(args, preprocessing)
    elif args.dataset is not None:
        dataset = DataSet(args.dataset)
        data = dataset.load()
        model = load_model(args, preprocessing)
    if args.expand_summary:
        model.summary(expand_nested=True)
    else:
        model.summary()
    if args.fig_dir is not None:
        img_dir = args.fig_dir
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if args.model:
            model_name = args.model
        else:
            model_name = f"{args.alpha}_MobileNet_{args.rho}"
        image_file_path = os.path.join(img_dir, model_name + ".png")
        plot_model(
            model,
            image_file_path,
            expand_nested=args.expand_summary,
        )
    return model, data


def main(args):
    model, dataset = summary_only(args)
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

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if x_train.shape[-1] == 1:
        raise ValueError("Gray scale images are not supported.")
    model.fit(
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
        help="Model to build. Required if you don't pass `rho` and `alpha`",
    )
    args.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="The alpha value for the model. Requuired if `model` not specified",
    )
    args.add_argument(
        "--rho",
        type=str,
        default="baseline",
        help="The rho value to choose. Requuired if `model` not specified",
        choices=list(RHO_TO_RES.keys()),
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
        "--fig_dir",
        type=str,
        help="The directory to save the figures",
        default=None,
    )
    args.add_argument(
        "--summary_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size for training",
    )
    args.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs for training",
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
    args.add_argument(
        "--expand_summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to expand the summary",
    )
    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()
    main(args)
