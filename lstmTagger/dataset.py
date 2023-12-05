import json
import urllib
import torch
import os
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



class TaggerDataset(IaiBaseDataset):
    def __init__(self, path: str, max_len) -> None:
        """
        Here we can load and preprocess the data.
        @param max_len: length of your sentences.
        @param path: you only need this parameter and it points to the root folder or file depending on your usecases.
        """
        super(TaggerDataset, self).__init__(path)

        self.tag_to_ix = {
            "DET": 0,
            "NN": 1,
            "V": 2,
            "PAD": 3,
        }  # Assign each tag with a unique index
        dict_path = path + "/tokenizer_dict.json"
        with smart_open.open(dict_path, "r", transport_params=_smart_open_transport_params(dict_path)) as f:
            self.to_ix = json.load(f)
        self.max_len = max_len

        self.x = []
        self.y = []
        data_path = path + "/data.csv"
        with smart_open.open(data_path, "r", transport_params=_smart_open_transport_params(data_path)) as f:
            for line in f.readlines():
                x, y = self.prepare_sequence(*line.strip().split(","))
                self.x.append(x)
                self.y.append(y)

    def __getitem__(self, item: int) -> torch.Tensor:
        return self.x[item], self.y[item]

    def __len__(self) -> int:
        return len(self.x)

    def prepare_sequence(self, seq, tags):
        """
        Convert words to padded tensors
        @param seq: words in the input sentence
        @param tags: words for labels
        @return: generated padded tensor ids
        """
        idxs = [self.to_ix[w.lower()] for w in seq.strip().split(" ")]
        idxs += [self.to_ix["PAD"]] * (self.max_len - len(idxs))
        tags = [self.tag_to_ix[w] for w in tags.strip().split(" ")]
        tags += [self.tag_to_ix["PAD"]] * (self.max_len - len(tags))
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(tags, dtype=torch.long)


if __name__ == "__main__":
    import os

    # specifying path this way allows it to be included in coverage
    this_path = os.path.abspath(f"{__file__}/../")

    def create_tokenizer_file():
        """
        The function we used once to generate the tokenizer dictionary.
        """
        data = []
        with open(f"{this_path}/sample_data/data.csv", "r") as f:
            for l in f.readlines():
                data.append(l.strip().split(",")[0])

        word_to_ix = {"PAD": 0}
        # For each words-list (sentence) and tags-list in each tuple of training_data
        for sent in data:
            for word in sent.split(" "):
                word = word.lower()
                if word not in word_to_ix:  # word has not been assigned an index yet
                    word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
        import json

        json.dump(word_to_ix, open(f"{this_path}/sample_data/tokenizer_dict.json", "w"))

    create_tokenizer_file()

    ds = TaggerDataset(f"{this_path}/sample_data", max_len=5)
    print(ds[0])
