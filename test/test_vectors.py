import torch
import os
from corefai.structs.vectors import LazyVectors

def test_lazy_vectors():
    VECTORS = LazyVectors(cache=os.path.expanduser('~/.vector_cache/'),
                            name='glove.6B.300d.txt',
                            vocab = {'hello', 'world'},
                                skim=None)
    weights = VECTORS.weights()
    embeddings = torch.nn.Embedding(weights.shape[0],
                                              weights.shape[1],
                                              padding_idx=0)
    embeddings.weight.data.copy_(weights)

    assert embeddings.weight.shape == (4, 300)

    VECTORS = LazyVectors(cache=os.path.expanduser('~/.vector_cache/'),
                            name='glove.6B.300d.txt',
                            vocab = None,
                                skim=None)
    weights = VECTORS.weights()
    embeddings = torch.nn.Embedding(weights.shape[0],
                                              weights.shape[1],
                                              padding_idx=0)
    embeddings.weight.data.copy_(weights)

    assert embeddings.weight.shape[0] > 10

    VECTORS = LazyVectors(cache=os.path.expanduser('~/.vector_cache/'),
                            name='glove.6B.300d.txt',
                            vocab = None,
                            vocab_file = 'data/vocab.txt',
                                skim=None)
    weights = VECTORS.weights()
    embeddings = torch.nn.Embedding(weights.shape[0],
                                              weights.shape[1],
                                              padding_idx=0)
    embeddings.weight.data.copy_(weights)

    assert embeddings.weight.shape[0] == 3