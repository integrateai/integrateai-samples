from integrate_ai_sdk.base_class import IaiBaseModule
import torch
from torch import nn


class LSTMTagger(IaiBaseModule):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
        """
        Here you should instantiate your layers based on the configs.
        @param embedding_dim: size of embedding output
        @param hidden_dim: size of lstm hidden state
        @param vocab_size: size of the tokenizer (total number of all possible words in the input)
        @param output_size: number of classes
        """

        # do not forget to call super init
        super(LSTMTagger, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(self.vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.

        # LSTM is not a valid DP module on its own, but will be replaced by DPLSTM module using Opacus
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.output_size)

    def forward(self, sentence):
        """
        the forward path of our model
        @param sentence: input tensor
        @return: the prediction tensor
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space.permute(0, 2, 1)


if __name__ == "__main__":
    # you can test your code here
    loss = nn.CrossEntropyLoss()
    t1 = torch.tensor([[0, 1, 2, 0, 1]])
    t2 = torch.tensor(
        [
            [0.3227, -0.1395, -0.2533],
            [0.2841, -0.1469, -0.2228],
            [0.3267, -0.1632, -0.2441],
            [0.3468, -0.1547, -0.2510],
            [0.3694, -0.1491, -0.2733],
        ]
    )
    t2 = t2.view(1, 3, 5)
    print(loss(t2, t1))
