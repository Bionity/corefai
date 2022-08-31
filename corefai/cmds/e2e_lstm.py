# -*- coding: utf-8 -*-

import argparse

from corefai.resolvers import E2E_LSTM_Resolver
from corefai.cmds.cmd import init


def main():
    parser = argparse.ArgumentParser(description='Create End-to-End Coreference Resolver based on LSTM encoding.')
    parser.set_defaults(Resolver=E2E_LSTM_Resolver)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train.')
    subparser.add_argument('--eval_interval', type=int, default=1, help='Evaluate every n epochs.')
    subparser.add_argument('--embeds_dim', type=int, default=400, help='Embeddings dimension.')
    subparser.add_argument('--hidden_dim', type=int, default=200, help='Hidden dimension.')
    subparser.add_argument('--vocab', type=str, default='data/vocab.txt', help='Vocabulary file.')
    subparser.add_argument('--glove_name', type=str, default='glove.6B.300d.txt', help='Glove embeddings file.')
    subparser.add_argument('--turian_name', type=str, default='hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt', help='Turian embeddings file.')
    subparser.add_argument('--char_filters', type=int, default=50, help='Number of character filters.')
    subparser.add_argument('--distance_dim', type=int, default=20, help='Distance dimension.')
    subparser.add_argument('--genre_dim', type=int, default=20, help='Genre dimension.')
    subparser.add_argument('--speaker_dim', type=int, default=20, help='Speaker dimension.')
    subparser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    subparser.add_argument('--mu', type=float, default=0.9, help=' coefficient used for computing running averages of gradient and its square.')
    subparser.add_argument('--nu', type=float, default=0.999, help=' coefficient used for computing running averages of gradient and its square')
    subparser.add_argument('--eps', type=float, default=1e-8, help='Epsilon.')
    subparser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    subparser.add_argument('--decay', type=float, default=0.99, help='Decay rate for learning rate.')
    subparser.add_argument('--decay_steps', type=int, default=10, help='Decay steps.')
    subparser.add_argument('--update_steps', type=int, default=1, help='Update steps.')
    subparser.add_argument('--amp', type=bool, help='If ``False``, disables gradient scaling. :meth:`step` simply')
    subparser.add_argument('--pattern', default='*conll', help='pattern of the files to train/evaluate')
    subparser.add_argument('--train', default='/data', help='path to train file')
    subparser.add_argument('--dev', default='/data', help='path to dev file')
    subparser.add_argument('--test', default='/data', help='path to test file')
    subparser.add_argument('--cache', default='~/.vectors_cache/', help='path to cache vectors file')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--data', default='/data', help='path to dataset')
    subparser.add_argument('--eval_script', default='../src/eval/scorer.pl', help='path to evaluation script')

    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--doc_path', default='/data/test.txt', help='path to test data for prediction')
    init(parser)


if __name__ == "__main__":
    main()