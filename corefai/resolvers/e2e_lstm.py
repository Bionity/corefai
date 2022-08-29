from corefai.utils import *
from corefai.utils.configs import Config
from corefai.models import E2E_LSTM
from corefai.resolvers import Resolver

class E2E_LSTM_Resolver(Resolver):
    """ Class dedicated to training and evaluating the model
    """

    NAME = "e2e-lstm-en"
    MODEL = E2E_LSTM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, num_epochs, eval_interval, train_corpus, val_corpus, **kwargs):
        return super().train(**Config().update(locals()))


    def evaluate(self, val_corpus, eval_script='../src/eval/scorer.pl'):
        """ Evaluate a corpus of CoNLL-2012 gold files """
        return super().evaluate(**Config().update(locals()))

    def predict(self, doc):
        """ Predict coreference clusters in a document """
        return super().predict(**Config().update(locals()))
    
