from corefai.utils import *
from corefai.utils.configs import Config
from corefai.models import E2E_LSTM
from corefai.resolvers import Resolver
from torch import optim
class E2E_LSTM_Resolver(Resolver):
    """ Class dedicated to training and evaluating the model
    """

    NAME = "e2e-lstm-en"
    MODEL = E2E_LSTM
    encoder = 'lstm'
    def __init__(self, 
        args: Config,

        **kwargs):
        super().__init__(args)
        self.args = args.update(locals())
        self.model = self.MODEL(**self.args)
        embeds_shapes = {'glove_shape0': self.model.encoder.glove_shape0, 
                            'glove_shape1': self.model.encoder.glove_shape1, 
                            'turian_shape0': self.model.encoder.turian_shape0, 
                                'turian_shape1': self.model.encoder.turian_shape1}
        self.args.update(embeds_shapes)
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
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

