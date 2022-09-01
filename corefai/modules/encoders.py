import torch
from torch import nn
from torch.nn import functional as F

from corefai.modules.cnn import CharCNN
from corefai.structs import LazyVectors
from corefai.utils.transforms import pack, unpack_and_unpad
from corefai.utils.tensor import lookup_tensor
from typing import Optional
import os

class LSTMDocumentEncoder(nn.Module):
    """ Document encoder for tokens
    """
    def __init__(
        self, 
        hidden_dim: int, 
        char_filters: int, 
        vocab: set = None, 
        glove_name: str = 'glove.6B.300d.txt',
        turian_name: str = 'hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt',
        glove_shape0: Optional[int] = 400000,
        glove_shape1: Optional[int] = 300,
        turian_shape0: Optional[int] = 400000,
        turian_shape1: Optional[int] = 50, 
        cache: str = os.path.expanduser('~/.vectors_cache/'),
        n_layers: int = 2,
        checkpoint: bool =False):
        super().__init__()
        if checkpoint:
            self.glove_shape0 = glove_shape0
            self.glove_shape1 = glove_shape1

            self.turian_shape0 = turian_shape0
            self.turian_shape1 = turian_shape1

            self.glove = nn.Embedding(self.glove_shape0, self.glove_shape1)
            self.turian = nn.Embedding(self.turian_weights0, self.turian_weights1)
        else:
            self.GLOVE = LazyVectors(vocab = vocab,
                                name=glove_name,
                                cache=cache)
            self.GLOVE.set_dicts()

            self.TURIAN = LazyVectors(vocab = vocab,
                                 name=turian_name,
                                 cache=cache)
            self.TURIAN.set_dicts()

            # Unit vector embeddings as per Section 7.1 of paper
            glove_weights = F.normalize(self.GLOVE.weights())
            turian_weights = F.normalize(self.TURIAN.weights())

            # GLoVE
            self.glove_shape0 = glove_weights.shape[0]
            self.glove_shape1 = glove_weights.shape[1]

            self.glove = nn.Embedding(self.glove_shape0, self.glove_shape1)
            self.glove.weight.data.copy_(glove_weights)
            self.glove.weight.requires_grad = False

            # Turian
            self.turian_shape0 = turian_weights.shape[0]
            self.turian_shape1 = turian_weights.shape[1]
        
            self.turian = nn.Embedding(self.turian_shape0, self.turian_shape1)
            self.turian.weight.data.copy_(turian_weights)
            self.turian.weight.requires_grad = False

        # Character
        self.char_embeddings = CharCNN(char_filters, vocab)

        # Sentence-LSTM
        self.lstm = nn.LSTM(self.glove_shape1+self.turian_shape1+char_filters,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout
        self.emb_dropout = nn.Dropout(0.50)
        self.lstm_dropout = nn.Dropout(0.20)

    def forward(self, doc):
        """ Convert document words to ids, embed them, pass through LSTM. """

        # Embed document
        embeds = [self.embed(s) for s in doc.sents]

        # Batch for LSTM
        packed, reorder = pack(embeds)

        # Apply embedding dropout
        self.emb_dropout(packed[0])

        # Pass an LSTM over the embeds
        output, _ = self.lstm(packed)

        # Apply dropout
        self.lstm_dropout(output[0])

        # Undo the packing/padding required for batching
        states = unpack_and_unpad(output, reorder)

        return torch.cat(states, dim=0), torch.cat(embeds, dim=0)

    def embed(self, sent):
        """ Embed a sentence using GLoVE, Turian, and character embeddings """

        # Embed the tokens with Glove
        glove_embeds = self.glove(lookup_tensor(sent, self.GLOVE))

        # Embed again using Turian this time
        tur_embeds = self.turian(lookup_tensor(sent, self.TURIAN))

        # Character embeddings
        char_embeds = self.char_embeddings(sent)

        # Concatenate them all together
        embeds = torch.cat((glove_embeds, tur_embeds, char_embeds), dim=1)

        return embeds
