from corefai.modules.encoders import LSTMDocumentEncoder
from corefai.modules.scores import MentionScore, PairwiseScore
from corefai.structs.dataset import Document, Span
from torch import nn 
import torch
from typing import Set, Any, Optional, List, Tuple

class E2E_LSTM(nn.Module):
    """ Super class to compute coreference links between spans
    """
    def __init__(self, embeds_dim: int = 400,
                       hidden_dim: int = 200,
                       vocab: Optional[Set[Any]] = None,
                       glove_name: str = 'glove.6B.300d.txt',
                       turian_name: str = 'hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt', 
                       cache: str = '/.vectors_cache/',
                       char_filters: int =50,
                       distance_dim: int =20,
                       genre_dim: int =20,
                       speaker_dim: int =20,
                       glove_shape0: int =400000,
                       glove_shape1:int =300,
                       turian_shape0=400000,
                       turian_shape1: int =50,
                       checkpoint: bool = False,
                       **kwargs):
        '''
        The model is based on using of word embeddings, character embeddings, encoded with LSTM
        Args:
            embeds_dim (int): dimension of word embeddings.
            hidden_dim (int): dimension of LSTM hidden state.
            vocab (Set[Any], optional): vocabulary of the model, set of words to take into account.
            glove_name (str, optional): name of the glove file, each file should be save in the cache folder, eg. /.vector_cache/glove.6B.300d.txt.
            turian_name (str, optional): name of the turian file, each file should be save in the cache folder, eg. /.vector_cache/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt.
            cache (str, optional): path to the cache folder.
            char_filters (int, optional): number of filters for character embeddings.
            distance_dim (int, optional): dimension of distance embeddings.
            genre_dim (int, optional): dimension of genre embeddings.
            speaker_dim (int, optional): dimension of speaker embeddings.
            glove_shape0 (int, optional): number of words in the glove file, if mentioned value is smaller than the actual number of words, the rest of the words will be ignored.
            glove_shape1 (int, optional): dimension of the glove embedding.
            turian_shape0 (int, optional): number of words in the turian file, if mentioned value is smaller than the actual number of words, the rest of the words will be ignored.
            turian_shape1 (int, optional): dimension of the turian embedding.
            checkpoint (bool, optional): if True, encoder will be uploaded from the checkpoint.
        '''
        super().__init__()
        # Forward and backward pass over the document
        attn_dim = hidden_dim*2

        # Forward and backward passes, avg'd attn over embeddings, span width
        gi_dim = attn_dim*2 + embeds_dim + distance_dim

        # gi, gj, gi*gj, distance between gi and gj
        gij_dim = gi_dim*3 + distance_dim + genre_dim + speaker_dim

        self.glove_name = glove_name
        self.turian_name = turian_name
        self.cache = cache

        # Initialize modules
        self.encoder = LSTMDocumentEncoder(
                                        hidden_dim = hidden_dim, 
                                        char_filters = char_filters, 
                                        vocab = vocab,
                                        glove_name = glove_name,
                                        turian_name = turian_name,  
                                        cache = cache,
                                        glove_shape0 = glove_shape0,
                                        glove_shape1 = glove_shape1,
                                        turian_shape0 = turian_shape0,
                                        turian_shape1 = turian_shape1,
                                        checkpoint = checkpoint,
                                        )
        self.score_spans = MentionScore(gi_dim, attn_dim, distance_dim)
        self.score_pairs = PairwiseScore(gij_dim, distance_dim, genre_dim, speaker_dim)

    def forward(self, doc: Document) -> Tuple[List[Span], torch.Tensor]:
        """ Enocde document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores
            Args:
                doc (Document): document to encode
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.encoder(doc)

        # Get mention scores for each span, prune
        spans, g_i, mention_scores = self.score_spans(states, embeds, doc)

        # Get pairwise scores for each span combo
        spans, coref_scores = self.score_pairs(spans, g_i, mention_scores)
        return spans, coref_scores