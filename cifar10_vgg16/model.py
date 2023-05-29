from torchvision import models
import torch.nn as nn
from integrate_ai_sdk.base_class import IaiBaseModule

# This is an example of using a pre-trained VGG model on CIFAR10 dataset using integrate_ai_sdk
# Download the data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and extract the file,
# to run the example on that dataset


class MyVgg(IaiBaseModule):
    def __init__(self):
        """
        We are not using any parameters in init so our json config is empty.
        """
        # Calling super init as this is a subclass
        super(MyVgg, self).__init__()

        # Adding VGG module to a ustom module and loading its weights from the net.
        # You can also have a file of weights and load your pre-trained module from that file.
        self.vgg = models.vgg16(pretrained=True)

        # Using vgg on my custom data so, the last layer is changed
        last_layer_in = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(last_layer_in, 10)

    def forward(self, x):
        """
        This is a simple pytorch module here
        :param x: input tensor
        :return: output of model
        """
        return self.vgg(x)


# Good practice to test your code here
if __name__ == "__main__":
    tmp_model = MyVgg()
