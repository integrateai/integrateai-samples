## Pre-trained VGG on CIFAR-10

This example uses a pre-trained VGG model on Imagenet to make predictions on cifar-10 images.

### Getting Started
Download data from [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and extract, in order to use it.

### Using integrate_ai_sdk

* To start we need to implement one class for models and one for dataset
* We created a custom model class `MyVgg` in `model.py` to use the pre-trained model
  * `MyVgg` model uses pre-trained vgg16 from pytorch and replaces it's classifier layer to match our data.
* A custom Dataset class `MyPickleLoader` in `dataset.py` to load the training data from pickle files.
  * You can download any other pre-trained weights and load it here.
  * We convert our label tensor type based on whether it is a classification task or regression task.
* `MyPickleLoader.json` and `MyVgg.json` are used for testing and contain the default inputs for custom dataset and custom model.
  * The default configs also enable other users to know how to structure their data to use your custom model
