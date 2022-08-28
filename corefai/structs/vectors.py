import torch

from torchtext.vocab import Vectors

from cached_property import cached_property

from corefai.utils.transforms import *
from corefai.utils.data import read_file

from typing import List, Set, Optional


class LazyVectors:
    """Load only those vectors from GloVE that are in the vocab if vocav is provided.
    Assumes PAD id of 0 and UNK id of 1
    """

    unk_idx = 1

    def __init__(self, name: str,
                       cache: str,
                       skim: Optional[int] = None,
                       vocab: Optional[Set[str]] = None,
                       vocab_file: Optional[str] = None):
        """  Requires the glove vectors to be in a folder named .vector_cache
        Setup:
            >> cd ~/where_you_want_to_save
            >> mkdir .vector_cache
            >> mv ~/where_glove_vectors_are_stored/glove.840B.300d.txt
                ~/where_you_want_to_save/.vector_cache/glove.840B.300d.txt
        Initialization (first init will be slow):
            >> VECTORS = LazyVectors(cache='~/where_you_saved_to/.vector_cache/',
                                     vocab_file='../path/vocabulary.txt',
                                     skim=None)
        Usage:
            >> weights = VECTORS.weights()
            >> embeddings = torch.nn.Embedding(weights.shape[0],
                                              weights.shape[1],
                                              padding_idx=0)
            >> embeddings.weight.data.copy_(weights)
            >> embeddings(sent_to_tensor('kids love unknown_word food'))
        You can access these moved vectors from any repository

        Args:
            name (str): name of the vector file
            cache (str): path to the cache directory
            skim (int): number of vectors to load
            vocab (set): set of words to load
        """
        self.__dict__.update(locals())
        if self.vocab is not None:
            self.set_vocab(vocab)
        elif self.vocab_file is not None:
            self.set_vocab(self.get_vocab(self.vocab_file))


    @classmethod
    def from_corpus(cls, corpus_vocabulary: set, name: str, cache: str):
        return cls(name=name, cache=cache, vocab=corpus_vocabulary)

    @cached_property
    def loader(self):
        return Vectors(self.name, cache=self.cache)

    def set_vocab(self, vocab: set):
        """ Set corpus vocab
        """
        # Intersects and initializes the torchtext Vectors class
        self.vocab = [v for v in vocab if v in self.loader.stoi][:self.skim]

        self.set_dicts()

    def get_vocab(self, filename: str):
        """ Read in vocabulary (top 30K words, covers ~93.5% of all tokens) """
        return read_file(filename)

    def set_dicts(self):
        """ _stoi: map string > index
            _itos: map index > string
        """
        if self.vocab is None:
            self._stoi = {s: i for i, s in enumerate(self.loader.stoi)}
            self._itos = {i: s for i, s in enumerate(self.loader.itos)}
        else:
            self._stoi = {s: i for i, s in enumerate(self.vocab)}
            self._itos = {i: s for s, i in self._stoi.items()}

    def weights(self):
        """Build weights tensor for embedding layer """
        # Select vectors for vocab words.
        if self.vocab is not None:
            weights = torch.stack([
                self.loader.vectors[self.loader.stoi[s]]
                for s in self.vocab
            ])
        else:
            weights = torch.stack([
                self.loader.vectors[self.loader.stoi[s]]
                for s in self.loader.itos
            ])

        # Padding + UNK zeros rows.
        return torch.cat([
            torch.zeros((2, self.loader.dim)),
            weights,
        ])

    def stoi(self, s: int):
        """ String to index (s to i) for embedding lookup """
        idx = self._stoi.get(s)
        return idx + 2 if idx else self.unk_idx

    def itos(self, i: int):
        """ Index to string (i to s) for embedding lookup """
        token = self._itos.get(i)
        return token if token else 'UNK'
