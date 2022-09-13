#the code is based on https://github.com/shayneobrien/coreference-resolution

import os, io, re, attr, random
from fnmatch import fnmatch
from copy import deepcopy as c
from cached_property import cached_property

from corefai.utils.transforms import flatten, compute_idx_spans
from corefai.utils.data import conll_clean_token


import nltk 

from typing import List, Any, Optional

CWD = '/'.join(os.getcwd().split('/')[:-1])

NORMALIZE_DICT = {"/.": ".", "/?": "?",
                  "-LRB-": "(", "-RRB-": ")",
                  "-LCB-": "{", "-RCB-": "}",
                  "-LSB-": "[", "-RSB-": "]"}
REMOVED_CHAR = ["/", "%", "*"]

@attr.s(frozen=True, repr=False)
class Span:
    """
    Object representing a span of tokens;
    """
    # Left / right token indexes
    i1 = attr.ib()
    i2 = attr.ib()

    # Id within total spans (for indexing into a batch computation)
    id = attr.ib()

    # Speaker
    speaker = attr.ib()

    # Genre
    genre = attr.ib()

    # Unary mention score, as tensor
    si = attr.ib(default=None)

    # List of candidate antecedent spans
    yi = attr.ib(default=None)

    # Corresponding span ids to each yi
    yi_idx = attr.ib(default=None)

    def __len__(self):
        return self.i2-self.i1+1

    def __repr__(self):
        return 'Span representing %d tokens' % (self.__len__())

class Document:
    def __init__(
        self, 
        raw_text: str, 
        tokens: List[str],
        lang: Optional[str] = 'en',
        corefs: Optional[List[Any]] = None, 
        speakers: Optional[str] = None, 
        genre: Optional[str] = None, 
        filename: Optional[str] = None):
        """
        Basic class for storing a document data.
        Args:
            raw_text (str): raw text of the document.
            tokens (str, list): list of tokens from the document.
            corefs (list): list of coreference chains.
            speakers (str, optional): speaker of the document.
            genre (str, optional): genre of the document.
            filename (str, optional): filename of the document.   
        """ 
        self.raw_text = raw_text
        self.tokens = tokens
        self.corefs = corefs
        self.speakers = speakers
        self.genre = genre
        self.filename = filename
        self.lang = lang

        if self.lang == 'en':
            self.punkt_sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        #TODO: add other languages
        else:
            self.punkt_sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # Filled in at evaluation time.
        self.tags = None

    def __getitem__(self, idx: int):
        return (self.tokens[idx], self.corefs[idx], \
                self.speakers[idx], self.genre)

    def __repr__(self):
        return 'Document containing %d tokens' % len(self.tokens)

    def __len__(self):
        return len(self.tokens)

    @cached_property
    def sents(self) -> List[List[str]]:
        """ Regroup raw_text into sentences 
            Returns:
                sents (list): list of sentences 
        """
        # Get sentences
        sent_idx = [sent for 
                            sent in self.punkt_sent_tokenizer.sentences_from_tokens(self.tokens)]
        return sent_idx

    def spans(self) -> List[Span]:
        """ Create Span object for each span """
        return [Span(i1=i[0], i2=i[-1], id=idx,
                    speaker=self.speaker(i), genre=self.genre)
                for idx, i in enumerate(compute_idx_spans(self.sents))]

    def truncate(self, MAX=50):
        """ Randomly truncate the document to up to MAX sentences """
        if len(self.sents) > MAX:
            i = random.sample(range(MAX, len(self.sents)), 1)[0]
            tokens = flatten(self.sents[i-MAX:i])
            return self.__class__(c(self.raw_text), tokens,
                                  c(self.corefs), c(self.speakers),
                                  c(self.genre), c(self.filename))
        return self

    def speaker(self, i):
        """ Compute speaker of a span """
        if self.speakers is None:
            return None
        elif len(self.speakers) <= i[0] or len(self.speakers) <= i[-1]:
            return None
        elif self.speakers[i[0]] == self.speakers[i[-1]]:
            return self.speakers[i[0]]
        return None


class Corpus:
    def __init__(
        self, 
        dirname: str,
        pattern: str = "*conll",
        dataset: str = 'CoNLL-2012'
        ):
        """
        Basic class for storing a corpus of documents.
        Args:
            dirname (str): path to the directory containing the corpus.
            pattern (str): pattern to match filenames.
            dataset (str): name of the dataset.
        """
        self.dirname = dirname
        self.pattern = pattern
        self.dataset = dataset
        self.documents = self.read_corpus()
        self.vocab, self.char_vocab = self.get_vocab()

    def __getitem__(self, idx):
        return self.documents[idx]

    def __repr__(self):
        return 'Corpus containg %d documents' % len(self.documents)

    def __len__(self):
        return len(self.documents)

    def get_vocab(self):
        """ Set vocabulary for LazyVectors """
        vocab, char_vocab = set(), set()

        for document in self.documents:
            vocab.update(document.tokens)
            char_vocab.update([char
                               for word in document.tokens
                               for char in word])

        return vocab, char_vocab

    def read_corpus(self):
        """Read coprus and returns list of Document objects"""
        files = self.parse_filenames()
        return flatten([self.load_file(file) for file in files])

    def load_file(self, filename:str):
        """ Load a *._conll file
            Args:
                filename (str): path to the file
            Returns:
                documents: list of Document class for each document in the file containing:
                tokens:                   split list of text
                utts_corefs:
                    coref['label']:     id of the coreference cluster
                    coref['start']:     start index (index of first token in the utterance)
                    coref['end':        end index (index of last token in the utterance)
                    coref['span']:      corresponding span
                utts_speakers:          list of speakers
                genre:                  genre of input
        """
        documents = []
        with io.open(filename, 'rt', encoding='utf-8', errors='strict') as f:
            raw_text, tokens, text, utts_corefs, utts_speakers, corefs, index = [], [], [], [], [], [], 0
            genre = filename.split('/')[6]
            for line in f:
                raw_text.append(line)
                cols = line.split()

                # End of utterance within a document: update lists, reset variables for next utterance.
                if len(cols) == 0:
                    if text:
                        tokens.extend(text), utts_corefs.extend(corefs), utts_speakers.extend([speaker]*len(text))
                        text, corefs = [], []
                        continue

                # End of document: organize the data, append to output, reset variables for next document.
                elif len(cols) == 2:
                    doc = Document(raw_text, tokens, utts_corefs, utts_speakers, genre, filename)
                    documents.append(doc)
                    raw_text, tokens, text, utts_corefs, utts_speakers, index = [], [], [], [], [], 0

                # Inside an utterance: grab text, speaker, coreference information.
                elif len(cols) > 7:
                    text.append(conll_clean_token(cols[3]))
                    speaker = cols[9]

                    # If the last column isn't a '-', there is a coreference link
                    if cols[-1] != u'-':
                        coref_expr = cols[-1].split(u'|')
                        for token in coref_expr:

                            # Check if coref column token entry contains (, a number, or ).
                            match = re.match(r"^(\(?)(\d+)(\)?)$", token)
                            label = match.group(2)

                            # If it does, extract the coref label, its start index,
                            if match.group(1) == u'(':
                                corefs.append({'label': label,
                                           'start': index,
                                           'end': None})

                            if match.group(3) == u')':
                                for i in range(len(corefs)-1, -1, -1):
                                    if corefs[i]['label'] == label and corefs[i]['end'] is None:
                                        break

                                # Extract the end index, include start and end indexes in 'span'
                                corefs[i].update({'end': index,
                                              'span': (corefs[i]['start'], index)})
                    index += 1
                else:
                    # Beginning of Document, beginning of file, end of file: nothing to scrape off
                    continue

        return documents

    def parse_filenames(self):
        """ Walk a nested directory to get all filename ending in a pattern """
        for path, subdirs, files in os.walk(self.dirname):
            for name in files:
                if fnmatch(name, self.pattern):
                    yield os.path.join(path, name)