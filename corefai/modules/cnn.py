import torch
from torch import nn
from torch.nn import functional as F

from corefai.utils.tensor import to_cuda

from typing import Optional, Set, List

class CharCNN(nn.Module):
    """ Character-level CNN. Contains character embeddings.
    """

    def __init__(self, filters:int, vocab: Optional[Set[str]] = None, char_dim: int = 8):
        """ Initialize a CharCNN model.
            Args:
                vocab (set): set of characters in the vocabulary
                filters (int): Number of channels produced by the convolution
                char_dim (int): dimension of the character embeddings
        """
        super().__init__()
        if vocab is None:
            self.vocab = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
                            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
                                    '-', '.', '\'', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        else:
            self.vocab = vocab
        self.unk_idx = 1
        self._stoi = {char: idx+2 for idx, char in enumerate(self.vocab)}
        self.pad_size = 15
        self.embeddings = nn.Embedding(len(self.vocab)+2, char_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.pad_size,
                                              out_channels=filters,
                                              kernel_size=n) for n in (3,4,5)])

    def forward(self, sent: List[str]) -> torch.Tensor:
        """ Compute filter-dimensional character-level features for each doc token 
            Args:
                sent (list): list of tokens
        """
        embedded = self.embeddings(self.sent_to_tensor(sent))
        convolved = torch.cat([F.relu(conv(embedded)) for conv in self.convs], dim=2)
        pooled = F.max_pool1d(convolved, convolved.shape[2]).squeeze(2)
        return pooled

    def sent_to_tensor(self, sent: List[str]) -> torch.Tensor:
        """ Batch-ify a document class instance for CharCNN embeddings 
            Args:
                sent (list): list of tokens
            
        """
        tokens = [self.token_to_idx(t) for t in sent]
        batch = self.char_pad_and_stack(tokens)
        return batch

    def token_to_idx(self, token: str) -> torch.Tensor:
        """ Convert a token to its character lookup ids 
            Args:
                token (str): token to convert.
        """
        return to_cuda(torch.tensor([self.stoi(c) for c in token]))

    def char_pad_and_stack(self, tokens: torch.Tensor) -> torch.Tensor:
        """ Pad and stack an uneven tensor of token lookup ids 
            Args:
                tokens (torch.Tensor): tensor of token lookup ids
        """
        skimmed = [t[:self.pad_size] for t in tokens]

        lens = [len(t) for t in skimmed]

        padded = [F.pad(t, (0, self.pad_size-length))
                  for t, length in zip(skimmed, lens)]

        return torch.stack(padded)

    def stoi(self, char: str) -> int:
        """ Convert a character to its lookup id;  <PAD> is 0, <UNK> is 1.
            Args:
                char (str): character to convert
        """
        idx = self._stoi.get(char)
        return idx if idx else self.unk_idx
