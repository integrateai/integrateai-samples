from typing import Tuple
import torch
from integrate_ai_sdk.base_class import IaiBaseDataset


class TemplateDataset(IaiBaseDataset):
    def __init__(self, path: str) -> None:
        """
        In this class you can load and pre-process the data.
        You can add additional params to do pre-processing or any transformations to your dataset
        @param path: path of your dataset - REQUIRED
        """
        super(TemplateDataset, self).__init__(path)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor]:
        """
        This method is responsible for producing each data point tensor.
        :param item:
        :return:
        """
        pass

    def __len__(self) -> int:
        """
        Returns the size of the dataset
        :return: dataset_size
        """
        pass


if __name__ == "__main__":
    dataset = TemplateDataset("path_to_your_sample_data")
