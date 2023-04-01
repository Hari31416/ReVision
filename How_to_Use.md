# How To Use

## Directory Structure

Each folder in the `ReVision` folder contains the implementation of a specific architecture. The `Notes` folder contains the notes for each architecture.

### `ReVision` Folder

There are five subfolders inside the `ReVision` folder. The `classification` has subfolders with their name the same as the architecture implemented in them. Each architecture subfolder contains a number of files that may vary from architecture to architecture. The main file, which contains the implementation of the architecture, is named `model.py`. The code is organized in a modular manner and model can be called using the `train.py` file with some command line arguments. These command line arguments may also vary from architecture to architecture. However, some of the arguments are common to all the architectures. A list of these arguments is given below:

- `--model`: The name of the model. This can be found because some models have multiple implementations. To get the list of available models, run the `train.py` file with the `--help` argument.
- `--dataset`: The dataset to be used for training. The dataset can be one of the following:
  - `mnist`: The MNIST dataset.
  - `cifar10`: The CIFAR-10 dataset.
  - `cifar100`: The CIFAR-100 dataset.
  - `fashion_mnist`: The Fashion MNIST dataset.
- `input_shape`: The input shape of the model. This is a tuple of three integers. The first integer is the number of channels, the second integer is the height of the image and the third integer is the width of the image. (For some models, it is required if `--dataset` is not specified. Or you just want to view a summary of the model by passing the `--summary_only` argument.)
- `--no_preprocess`: Whether to NOT use a preprocessin layer for the model.
- `--batch_size`: The batch size to be used for training.
- `--epochs`: The number of epochs to be used for training.
- `--summary_only`: A boolean value. If set to `True`, only the summary of the model will be printed.
- `--fig_directory`: The directory in which the summary of the model will be saved. The name used will be the name of the model. (Only for a tensorflow model.)
- `--lr`: The learning rate to be used for training.
- `--optimizer`: The optimizer to be used for training. The optimizer can be one of the following:
  - `sgd`: The Stochastic Gradient Descent optimizer.
  - `adam`: The Adam optimizer.
  - `rmsprop`: The RMSProp optimizer.
  - `adagrad`: The AdaGrad optimizer.
  - `nadam`: The NAdam optimizer.
  - `adamax`: The Adamax optimizer.
- `--expand_summary`: A boolean value. If set to `True`, the summary of the model will be expanded if possible.

The `generative` subfolder has some implementation related to generative models and `detection` has some implementation related to object detection. See the `Notes` folder to see how to use them.

### `Notes` Folder

The `Notes` folder contains the notes for each architecture. The notes are written in markdown format where the file name is the name of the architecture. For example, if you want to read the notes for the LeNet architecture, read the `LeNet.md` file in the `Notes` folder.

## Calling the Model

To call and train a model, create a new file in the root folder and write the following code:

```python
# Description: Example of how to use an implementation of a model

# First, import the train.py module for that model. In this case, we are using LeNet
from ReVision.classification.LeNet import train

if __name__ == "__main__":
    # Then, we can call the arg_parse function to get the arguments for the model
    args = train.arg_parse()
    # Finally, we can call the main function to train the model
    train.main(args)

```

Or just modify the `example.py` file in the root folder.

This is an example of using the architecures in `classification` subfolder. Others are somewhat similar. See the `Notes` folder for more information.
