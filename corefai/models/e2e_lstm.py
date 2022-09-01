from corefai.modules.encoders import LSTMDocumentEncoder
from corefai.modules.scores import MentionScore, PairwiseScore
from torch import nn 

class E2E_LSTM(nn.Module):
    """ Super class to compute coreference links between spans
    """
    def __init__(self, embeds_dim = 400,
                       hidden_dim = 200,
                       vocab = None,
                       glove_name = 'glove.6B.300d.txt',
                       turian_name = 'hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt', 
                       cache = '/.vectors_cache/',
                       char_filters=50,
                       distance_dim=20,
                       genre_dim=20,
                       speaker_dim=20,
                       glove_shape0=400000,
                       glove_shape1=300,
                       turian_shape0=400000,
                       turian_shape1=50,
                       checkpoint = False,
                       **kwargs):

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

    def forward(self, doc):
        """ Enocde document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.encoder(doc)

        # Get mention scores for each span, prune
        spans, g_i, mention_scores = self.score_spans(states, embeds, doc)

        # Get pairwise scores for each span combo
        spans, coref_scores = self.score_pairs(spans, g_i, mention_scores)
        return spans, coref_scores