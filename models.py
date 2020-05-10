import numpy as np
import torch.nn as nn


# ----------------------------------------------------------------------------
def initialize_xavier(module):
    """
    Initializes convolutional and linear module weights according to
    `Xavier <http://pytorch.org/docs/stable/nn.html#torch.nn.init.xavier_uniform_>`_
    and sets biases to an arbitrary small value close to zero

    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch module (e.g., torch.nn.Linear)
    """
    # filter according to the module type
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        # initialize module weight with xavier uniform
        nn.init.xavier_uniform_(module.weight)
        # initialize module bias with small values close to zero
        if module.bias is not None:
            module.bias.data.fill_(0.01)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
def count_parameters(module):
    """
    Counts the trainable parameters of the class PyTorch model

    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch model (e.g., Net)

    Returns
    -------
    int
        The model's trainable parameter number
    """
    # calculate and return the number of trainable parameters
    return np.sum(p.numel() for p in module.parameters() if p.requires_grad)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
class Flatten(nn.Module):
    """
    PyTorch layer that flattens tensors
    """

    def __init__(self):
        # initialize parent class variables
        super(Flatten, self).__init__()

    def forward(self, thick_tensor):
        # flatten the input tensor
        flat_tensor = thick_tensor.view(thick_tensor.size()[0], -1)
        # return the flatten input tensor
        return flat_tensor
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
class Net(nn.Module):
    """
    PyTorch model that predicts facial landmarks
    """

    def __init__(self):
        # initialize parent class variables
        super(Net, self).__init__()

        # Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)

        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers,
        # and other layers (such as dropout or batch normalization) to avoid overfitting

        # define the feature extractor backbone
        self.extractor = nn.Sequential(nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2, bias=False),
                                       nn.BatchNorm2d(8),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=False),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       Flatten())

        # define the keypoint regressor
        self.regressor = nn.Sequential(nn.Linear(512*2*2, 136))

    def forward(self, x):
        # Define the feed forward behavior of this model
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))

        # extract features from the input image
        x = self.extractor(x)
        # estimate the facial landmarks from image features
        x = self.regressor(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
# ----------------------------------------------------------------------------
