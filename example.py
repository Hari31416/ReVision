# Description: Example of how to use a n implementation of a model

# FIrst, import the train.py module for that model. In this case, we are using LeNet
from ReVision.LeNet import train

if __name__ == "__main__":
    # Then, we can call the arg_parse function to get the arguments for the model
    args = train.arg_parse()
    # Finally, we can call the main function to train the model
    train.main(args)
    # Great! Train the model.
