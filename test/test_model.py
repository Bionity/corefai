from corefai.models import E2E_LSTM
from corefai.structs import Document
import os

def test_e2e_lstm():
    """ Test mention score module """

    distance_dim = 20
    embeds_dim=400
    hidden_dim=200

    attn_dim = hidden_dim*2
    # Forward and backward passes, avg'd attn over embeddings, span width
    gi_dim = attn_dim*2 + embeds_dim + distance_dim

    text = "Hello world! This is a test."
    tokens = ['Hello', 'world', '!', 'This', 'is', 'a', 'test', '.']
    doc = Document(text, tokens)

    scorer = E2E_LSTM(embeds_dim, hidden_dim, distance_dim = distance_dim,
        glove_name = 'glove.6B.300d.txt', 
        turian_name = 'hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt', 
        cache = os.path.expanduser('~/.vector_cache/'))

    spans, cored_scores = scorer(doc)
