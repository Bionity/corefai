
from corefai.resolvers import (E2E_LSTM_Resolver, Resolver)
from corefai.structs import (Corpus, Document, Span)
from corefai.models import (E2E_LSTM)

__all__ = ['E2E_LSTM_Resolver',
           'Resolver',
           'Corpus',
           'Dataset',
           'Span',
           'E2E_LSTM']

__version__ = '0.0.1'

RESOLVER = {resolver.NAME: resolver for resolver in [E2E_LSTM_Resolver]}
SRC = {'gcp': 'corefai-models'}
NAME = {
    'e2e-lstm-en': 'e2e_lstm_en',
}
MODEL = {src: {n: f"{link}/{m}.zip" for n, m in NAME.items()} for src, link in SRC.items()}

