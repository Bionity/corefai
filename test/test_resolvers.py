from corefai.resolvers import Resolver, E2E_LSTM_Resolver
from corefai.utils.configs import Config
import os

def test_resolver():
    distance_dim = 20
    embeds_dim = 400
    hidden_dim = 200

    args = Config(
        embeds_dim = embeds_dim,
        hidden_dim = hidden_dim,
        glove_name = 'glove.6B.300d.txt',
        turian_name = 'hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt',
        cache = os.path.expanduser('~/.vector_cache/'),
        distance_dim = distance_dim,
        pattern = '*conll',
        lr = 0.001,
        mu = 0.9,
        nu = 0.999,
        eps = 1e-8,
        weight_decay = 0,
        decay = 0.99,
        decay_steps = 10,
        amp = False)
        

    resolver = E2E_LSTM_Resolver(args)
    resolver.train(
                num_epochs=3,
                eval_interval=1,
                train_corpus = 'data/train',
                val_corpus = 'data/development',
                )
