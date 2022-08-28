from corefai.structs import Document, Corpus

def test_dataset():
    text = "Hello world! This is a test."
    tokens = ['Hello', 'world', '!', 'This', 'is', 'a', 'test', '.']
    doc = Document(text, tokens)
    assert doc.lang =='en'
    assert doc.corefs == None
    assert doc.speakers == None
    assert doc.genre == None
    assert doc.filename == None
    sents = doc.sents
    assert sents == [['Hello', 'world', '!'], ['This', 'is', 'a', 'test', '.']]

    spans = doc.spans()
    assert len(spans) == 21

    doc.truncate()

def test_corpus():
    corpus = Corpus(dirname = 'data', pattern = '*conll')
    assert len(corpus) > 0
    assert len(corpus.vocab) > 0
    assert len(corpus.char_vocab) > 0