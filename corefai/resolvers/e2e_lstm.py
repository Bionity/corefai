from corefai.utils import *
from corefai.utils.configs import Config
from corefai.models import E2E_LSTM
from corefai.resolvers import Resolver

class E2E_LSTM_Resolver(Resolver):
    """ Class dedicated to training and evaluating the model
    """

    NAME = "e2e-lstm-en"
    MODEL = E2E_LSTM
    encoder = 'lstm'
    def __init__(self, 
        args: Config,
        # embeds_dim = 400,
        # hidden_dim = 200,
        # vocab = None,
        # glove_name = 'glove.6B.300d.txt',
        # turian_name = 'hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt', 
        # cache = '/.vectors_cache/',
        # char_filters=50,
        # distance_dim=20,
        # genre_dim=20,
        # speaker_dim=20,
        **kwargs):
        super().__init__(args, **kwargs)
        self.args = args.update(locals())
        self.model = self.MODEL(args.embeds_dim, args.hidden_dim, args.vocab, args.glove_name, args.turian_name, 
                                        args.cache, args.char_filters, args.distance_dim, args.genre_dim, args.speaker_dim)
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.epoch = 0

    def train(self, num_epochs, eval_interval, train_corpus, val_corpus, **kwargs):
        return super().train(**Config().update(locals()))


    def evaluate(self, val_corpus, eval_script='../src/eval/scorer.pl'):
        """ Evaluate a corpus of CoNLL-2012 gold files """
        return super().evaluate(**Config().update(locals()))

    def predict(self, doc):
        """ Predict coreference clusters in a document """
        return super().predict(**Config().update(locals()))
    
