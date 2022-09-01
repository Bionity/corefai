# -*- coding: utf-8 -*-

# originated from https://github.com/yzhangcs/parser/blob/main/supar/utils/config.py

import argparse
import os
from ast import literal_eval
from configparser import ConfigParser

import corefai
from corefai.utils.data import download


class Config(object):

    def __init__(self, **kwargs):
        super(Config, self).__init__()

        self.update(kwargs)

    def __repr__(self):
        s = line = "-" * 20 + "-+-" + "-" * 30 + "\n"
        s += f"{'Param':20} | {'Value':^30}\n" + line
        for name, value in vars(self).items():
            s += f"{name:20} | {str(value):^30}\n"
        s += line

        return s

    def __getitem__(self, key):
        return getattr(self, key)

    def __getattr__(self, item):
        return None

    def __contains__(self, key):
        return hasattr(self, key)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

    def update(self, kwargs):
        for key in ('self', 'cls', '__class__'):
            kwargs.pop(key, None)
        kwargs.update(kwargs.pop('kwargs', dict()))
        for name, value in kwargs.items():
            setattr(self, name, value)
        return self

    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default

    def pop(self, key, val=None):
        return self.__dict__.pop(key, val)

    @classmethod
    def load(cls, conf='', unknown=None, **kwargs):
        config = ConfigParser()
        try:
            config.read(conf)
        except:
            raise ValueError(f"Config file {conf} not found")
        config = dict((name, literal_eval(value))
                      for section in config.sections()
                      for name, value in config.items(section))
        if unknown is not None:
            parser = argparse.ArgumentParser()
            for name, value in config.items():
                parser.add_argument('--'+name.replace('_', '-'), type=type(value), default=value)
            config.update(vars(parser.parse_args(unknown)))
        config.update(kwargs)
        return cls(**config)
