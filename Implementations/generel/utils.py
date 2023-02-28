import tensorflow as tf
from tensorflow.keras.optimizers import (
    Adam,
    SGD,
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    Nadam,
)

ALL_LOSSES = ["binary_crossentropy", "categorical_crossentropy", "mean_squared_error"]
ALL_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "auc",
    "mean",
    "mean_absolute_error",
    "mean_squared_error",
]


class Optimizer:
    def __init__(self, lr) -> None:
        self.lr = lr
        self.all_optimizers = {
            "adam": Adam,
            "sgd": SGD,
            "rmsprop": RMSprop,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "nadam": Nadam,
        }
        self.optimizer = None

    def get_optimizer(self, name):
        if name not in self.all_optimizers.keys():
            raise ValueError(f"Optimizer {name} is not supported")

        self.optimizer = self.all_optimizers[name](learning_rate=self.lr)
        return self.optimizer


def return_optimizer(optimizer, lr):
    return Optimizer(lr).get_optimizer(optimizer)


def return_loss(loss):
    if loss not in ALL_LOSSES:
        raise ValueError(
            f"Loss {loss} is not supported. Supported losses are: {ALL_LOSSES}"
        )
    return loss


def return_metric(metric):
    if metric not in ALL_METRICS:
        raise ValueError(
            f"Metric {metric} is not supported. Supported metrics are: {ALL_METRICS}"
        )
    return metric


def plot_model(model, fig_dir, **kwargs):
    tf.keras.utils.plot_model(
        model,
        to_file=fig_dir,
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        **kwargs,
    )
