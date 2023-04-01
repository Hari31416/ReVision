import argparse
import os
from ReVision.generative.style_transfer import StyleTransfer


def main(args):
    st = StyleTransfer(
        content_image_path=args.content_image_path,
        style_image_path=args.style_image_path,
    )

    if args.display_original_images:
        st.display_images()

    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    image_frequency = args.image_frequency
    notebook = False
    alpha = args.alpha
    beta = args.beta
    show_images = args.show_images
    lr = args.lr

    images = st.stylize_image(
        alpha=alpha,
        beta=beta,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        image_frequency=image_frequency,
        notebook=notebook,
        show_images=show_images,
        lr=lr,
    )

    if args.animate:
        name = (
            args.animation_name
            if args.animation_name is not None
            else "style_transfer.mp4"
        )
        if "." not in name:
            name += ".mp4"
        duration = args.duration if args.duration is not None else 20
        st.make_animation(images, name, duration)


def arg_parse():
    args = argparse.ArgumentParser(add_help=True)
    args.add_argument(
        "--content_image_path",
        type=str,
        default=os.path.join(
            os.path.expanduser("~"),
            ".keras/datasets/content_image.jpg",
        ),
        help="Path to the content image",
    )
    args.add_argument(
        "--style_image_path",
        type=str,
        default=os.path.join(
            os.path.expanduser("~"),
            ".keras/datasets/style_image.jpg",
        ),
        help="Path to the style image",
    )
    args.add_argument(
        "--display_original_images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display the original images",
    )
    args.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
    )
    args.add_argument(
        "--steps_per_epoch",
        type=int,
        default=10,
        help="Number of steps per epoch",
    )
    args.add_argument(
        "--lr",
        type=float,
        default=40.0,
        help="Initial learning rate for the optimizer. (Use large value ~20)",
    )
    args.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="Weight of the content loss",
    ),
    args.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Weight of the style loss",
    )
    args.add_argument(
        "--show_images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the images",
    )
    args.add_argument(
        "--image_frequency",
        type=int,
        default=2,
        help="Frequency of displaying the images",
    )
    args.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Animate the images",
    )
    args.add_argument(
        "--animation_name",
        type=str,
        default=None,
        help="Name of the animation",
    )
    args.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration of the animation",
    )

    args = args.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parse()
    main(args)
