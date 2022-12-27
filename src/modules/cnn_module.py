import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CNNModule(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(CNNModule, self).__init__()

        w = flatten(*flatten(*flatten(w=input_shape[1], k=3, s=1, p=0, m=True)))[0]
        h = flatten(*flatten(*flatten(w=input_shape[2], k=3, s=1, p=0, m=True)))[0]

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 64, 3)
        self.max_pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = (nn.Linear(w*h*64, output_shape))

    def forward(self, inputs):
        outputs = F.relu(self.max_pool1(self.conv1(inputs)))
        outputs = F.relu(self.max_pool2(self.conv2(outputs)))
        outputs = F.relu(self.max_pool3(self.conv3(outputs)))
        outputs = F.relu(self.fc1(self.flatten(outputs)))
        return outputs


def flatten(w, k=3, s=1, p=0, m=True):
    """
    Returns the right size of the flattened tensor after
        convolutional transformation
    :param w: width of image
    :param k: kernel size
    :param s: stride
    :param p: padding
    :param m: max pooling (bool)
    :return: proper shape and params: use x * x * previous_out_channels

    Example:
    r = flatten(*flatten(*flatten(w=100, k=3, s=1, p=0, m=True)))[0]
    self.fc1 = nn.Linear(r*r*128, 1024)
    """
    return int((np.floor((w - k + 2 * p) / s) + 1) / 2 if m else 1), k, s, p, m
