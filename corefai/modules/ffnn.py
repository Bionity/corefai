import torch
import torch.nn as nn

from corefai.utils.tensor import to_cuda

from typing import List

class Score(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim: int, hidden_dim: int=150):
        """ Initialize the scoring module
            Args:
                embeds_dim (int): dimension of the input embeddings
                hidden_dim (int): dimension of the hidden layer
        """
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)


class Distance(nn.Module):
    """ Learned, continuous representations for: span widths, distance
    between spans
    """

    bins = [1,2,3,4,8,16,32,64]

    def __init__(self, distance_dim=20):
        """ Initialize the distance module
            Args:
                distance_dim (int): dimension of the distance representation
        """
        super().__init__()

        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, distance_dim),
            nn.Dropout(0.20)
        )

    def forward(self, *args):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def stoi(self, lengths: torch.Tensor) -> torch.Tensor:
        """ Find which bin a number falls into """
        return to_cuda(torch.tensor([
            sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False
        ))


class Genre(nn.Module):
    """ 
        Learned continuous representations for genre. Zeros if genre unknown.
    """

    genres = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
    _stoi = {genre: idx+1 for idx, genre in enumerate(genres)}

    def __init__(self, genre_dim=20):
        """ Initialize the genre module
            Args:
                genre_dim (int): dimension of the genre representation
        """
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(len(self.genres)+1, genre_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, labels):
        """ Embedding table lookup """
        return self.embeds(self.stoi(labels))

    def stoi(self, labels: List[str]):
        """ Locate embedding id for genre 
            Args:
                labels (List[str]): list of genre labels;
                    e.g. [None, 'bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
            Returns:
                torch.Tensor: embedding id for each genre label
        """
        indexes = [self._stoi.get(gen) for gen in labels]
        return to_cuda(torch.tensor([i if i is not None else 0 for i in indexes]))

class Speaker(nn.Module):
    """ Learned continuous representations for binary speaker. Zeros if speaker unknown.
    """

    def __init__(self, speaker_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(3, speaker_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, speaker_labels):
        """ Embedding table lookup (see src.utils.speaker_label fnc) """
        return self.embeds(to_cuda(torch.tensor(speaker_labels)))

