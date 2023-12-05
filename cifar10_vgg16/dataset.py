import urllib

import torch
import numpy as np
from torchvision.transforms import transforms

from integrate_ai_sdk.base_class import IaiBaseDataset
import smart_open


def _smart_open_transport_params(filepath):
    sr = urllib.parse.urlsplit(filepath, scheme="")
    out = None
    if sr.scheme == "azure":
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        # THIS IS AN EXAMPLE - use your own method to supply the storage account
        storage_account = os.environ.get("IAI_AZURE_BLOB_STORAGE_ACCOUNT")
        account_url = f"https://{storage_account}.blob.core.windows.net"
        if not storage_account:
            raise Exception(
                "Unable to use azure blob storage without providing IAI_AZURE_BLOB_STORAGE_ACCOUNT env var."
            )
        out = {"client": BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())}
    return out


# This is an example of using a pre-trained VGG model on CIFAR10 dataset using integrate_ai_sdk
# Download the data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and extract the file,
# to run the example on that dataset


class MyPickleLoader(IaiBaseDataset):
    """
    A custom dataset class which is going to load a batch pickle data
    """

    def __init__(self, path, normalize_values=None):
        """
        init function should always gets at least one parameter which is `path`.
        More parameters can be added if you want to customize the json config.
        For example here a parameter is added to apply additional transformations based on the config.
        :param path: Load data from this path.
        :param normalize_values: Normalize images before feeding them to the models
        """
        # Calling super init as this is a subclass
        super(MyPickleLoader, self).__init__(path)

        self.data = MyPickleLoader._unpickle(path)
        # Good practice to set your labels to long if it is classification, or to float if it is regression
        self.label = torch.Tensor(self.data[b"labels"]).long()
        self.data = self.data[b"data"].reshape((-1, 3, 32, 32))  # Convert data to desired shape
        self.data = np.transpose(self.data, (0, 2, 3, 1))  # PIL needs the images to be channel last

        print("Data Loaded... shape:", self.data.shape)

        # Any transformations can be performed here
        transform = list()
        transform.append(transforms.ToPILImage())  # From numpy.ndarray to PIL Image!
        transform.append(transforms.Resize(size=(224, 224)))  # VGG only accepts 224*224
        transform.append(transforms.ToTensor())  # Datasets return tensors
        if normalize_values:
            transform.append(transforms.Normalize(*normalize_values))
        self.transform = transforms.Compose(transform)

    @staticmethod
    def _unpickle(file):
        import pickle

        with smart_open.open(file, "rb", transport_params=_smart_open_transport_params(file)) as f:
            unpickle_dict = pickle.load(f, encoding="bytes")
        return unpickle_dict

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item: int):
        return self.transform(self.data[item]), self.label[item]


# Good practice to test your code here
if __name__ == "__main__":
    default_normalize = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    ds = MyPickleLoader(
        path="......../cifar-10-batches-py/data_batch_1",
        normalize_values=default_normalize,
    )
    print(ds[0][0].shape)
