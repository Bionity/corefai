from corefai.resolvers import Resolver
from corefai.models import E2E_LSTM

def test_loading():
    resolver = Resolver.load_model('ckpts/e2e_lstm')
    assert resolver.NAME == 'e2e-lstm-en'
    assert resolver.MODEL == E2E_LSTM