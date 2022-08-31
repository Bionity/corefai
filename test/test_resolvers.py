from corefai.resolvers import Resolver
from corefai.models import E2E_LSTM
from corefai.structs import Document, Corpus
import os

def test_resolver():
    train_corpus = Corpus(dirname = 'data/train', pattern = '*conll')
    val_corpus = Corpus(dirname = 'data/development', pattern = '*conll')
    distance_dim = 20
    embeds_dim = 400
    hidden_dim = 200
    scorer = E2E_LSTM(embeds_dim, hidden_dim, distance_dim = distance_dim,
        glove_name = 'glove.6B.300d.txt', 
        turian_name = 'hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt', 
        cache = os.path.expanduser('~/.vector_cache/'))

    resolver = Resolver(scorer)
    resolver.MODEL = E2E_LSTM
    resolver.NAME = 'e2e-lstm-en'
    resolver.train(
                num_epochs=3,
                eval_interval=1,
                train_corpus=train_corpus,
                val_corpus=val_corpus,
                )