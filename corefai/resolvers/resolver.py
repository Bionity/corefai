import torch
import torch.optim as optim

import os
import io
import random
import dill
import json

import numpy as np
import networkx as nx
from tqdm import tqdm
from datetime import datetime
from subprocess import Popen, PIPE
from torch.cuda.amp import GradScaler
from typing import Dict, Any, Optional

import corefai
from corefai.utils.transforms import safe_divide, extract_gold_corefs, flatten
from corefai.utils.tensor import to_cuda
from corefai.utils.data import download, CACHE
from corefai.utils.configs import Config
from corefai.structs import Document, Corpus


class Resolver:
    """ Class dedicated to training and evaluating the model
    """

    NAME = None
    MODEL = None
    encoder = None
    def __init__(
            self,
            args: Config,
            **kwargs
            ):
        self.model = self.MODEL(**args)
        
        self.args = args.update(locals())

        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.epoch = 0

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(
            self, 
            num_epochs: int, 
            eval_interval: int, 
            train_corpus: str, val_corpus:str, **kwargs):
        """ Train a model """
        args = self.args.update(locals())

        train_corpus = Corpus(dirname = args.train_corpus, pattern = args.pattern)
        val_corpus = Corpus(dirname = args.val_corpus, pattern = args.pattern)

        train_corpus = [doc for doc in list(train_corpus) if doc.sents]
        val_corpus = [doc for doc in list(val_corpus) if doc.sents]
        
        if self.optimizer is None and self.scheduler is None:
            if self.encoder == 'lstm':
                self.optimizer = optim.Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
                self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
            elif self.encoder == 'transformer':
                self.optimizer = optim.Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
                self.scheduler = optim.lr_scheduler.InverseSquareRootLR(self.optimizer, args.warmup_steps)
            else:
                steps = len(train_corpus) * num_epochs // args.update_steps
                self.optimizer = optim.AdamW(
                    [{'params': p, 'lr': args.lr * (1 if n.startswith('encoder') else args.lr_rate)}
                    for n, p in self.model.named_parameters()],
                    args.lr,
                    (args.mu, args.nu),
                    args.eps,
                    args.weight_decay
                )
                self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, int(steps*args.warmup), steps)

        self.scaler = GradScaler(enabled=args.amp)

        if not os.path.exists("ckpts/"):
            os.makedirs("ckpts/")
        
        for epoch in range(1, args.num_epochs+1):
            self.epoch = epoch

            self.train_epoch(epoch, train_corpus, steps=100)

            # Save often
            self.save_model("ckpts/{}".format(str(datetime.now())))

            # Evaluate every eval_interval epochs
            if epoch % eval_interval == 0:
                print('\n\nEVALUATION\n\n')
                self.model.eval()
                results = self.evaluate(val_corpus)
                print(results)

    def train_epoch(self, epoch, train_corpus, steps):
        """ Run a training epoch over 'steps' documents """

        # Set model to train (enables dropout)
        self.model.train()

        # Randomly sample documents from the train corpus
        batch = random.sample(train_corpus, steps)

        epoch_loss, epoch_mentions, epoch_corefs, epoch_identified = [], [], [], []

        for document in tqdm(batch):
            # Randomly truncate document to up to 50 sentences
            doc = document.truncate()

            # Compute loss, number gold links found, total gold links
            try:
                loss, mentions_found, total_mentions, \
                corefs_found, total_corefs, corefs_chosen = self.train_doc(doc)
            except:
                continue

            # Track stats by document for debugging
            print(document, '| Loss: %f | Mentions: %d/%d | Coref recall: %d/%d | Corefs precision: %d/%d' \
                % (loss, mentions_found, total_mentions,
                    corefs_found, total_corefs, corefs_chosen, total_corefs))

            epoch_loss.append(loss)
            epoch_mentions.append(safe_divide(mentions_found, total_mentions))
            epoch_corefs.append(safe_divide(corefs_found, total_corefs))
            epoch_identified.append(safe_divide(corefs_chosen, total_corefs))

        print('Epoch: %d | Loss: %f | Mention recall: %f | Coref recall: %f | Coref precision: %f' \
                % (epoch, np.mean(epoch_loss), np.mean(epoch_mentions),
                    np.mean(epoch_corefs), np.mean(epoch_identified)))

    def train_doc(self, document):
        """ Compute loss for a forward pass over a document """
        # Extract gold coreference links
        gold_corefs, total_corefs, \
            gold_mentions, total_mentions = extract_gold_corefs(document)

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Init metrics
        mentions_found, corefs_found, corefs_chosen = 0, 0, 0

        # Predict coref probabilites for each span in a document
        spans, probs = self.model(document)

        # Get log-likelihood of correct antecedents implied by gold clustering
        gold_indexes = to_cuda(torch.zeros_like(probs))
        for idx, span in enumerate(spans):

            # Log number of mentions found
            if (span.i1, span.i2) in gold_mentions:
                mentions_found += 1

                # Check which of these tuples are in the gold set, if any
                golds = [
                    i for i, link in enumerate(span.yi_idx)
                    if link in gold_corefs
                ]

                # If gold_pred_idx is not empty, consider the probabilities of the found antecedents
                if golds:
                    gold_indexes[idx, golds] = 1

                    # Progress logging for recall
                    corefs_found += len(golds)
                    found_corefs = sum((probs[idx, golds] > probs[idx, len(span.yi_idx)])).detach()
                    corefs_chosen += found_corefs.item()
                else:
                    # Otherwise, set gold to dummy
                    gold_indexes[idx, len(span.yi_idx)] = 1

        # Negative marginal log-likelihood
        eps = 1e-8
        loss = torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp_(eps, 1-eps)) * -1)

        # Backpropagate
        loss.backward()

        # Step the optimizer
        self.optimizer.step()
        
        # Step the learning rate decrease scheduler
        self.scheduler.step()


        return (loss.item(), mentions_found, total_mentions,
                corefs_found, total_corefs, corefs_chosen)

    def evaluate(self, val_corpus, eval_script='eval/scorer.pl'):
        """ Evaluate a corpus of CoNLL-2012 gold files """

        args = self.args.update(locals())

        if isinstance(val_corpus, str):
            val_corpus = Corpus(dirname = args.val_corpus, pattern = args.pattern)
    
        # Predict files
        print('Evaluating on validation corpus...')
        predicted_docs = [self.predict(doc) for doc in tqdm(val_corpus)]
        val_corpus.docs = predicted_docs

        # Output results
        golds_file, preds_file = self.to_conll(val_corpus, eval_script)

        # Run perl script
        print('Running Perl evaluation script...')
        p = Popen([eval_script, 'all', golds_file, preds_file], stdout=PIPE)
        stdout, stderr = p.communicate()
        results = str(stdout).split('TOTALS')[-1]

        if not os.path.exists("preds/"):
            os.makedirs("preds/")

        # Write the results out for later viewing
        with open('preds/results.txt', 'w+') as f:
            f.write(results)
            f.write('\n\n\n')

        return results

    def predict(self, doc: Optional[Document] = None, doc_path: Optional[str] = None):
        """ Predict coreference clusters in a document """
        args = self.args.update(locals())

        # Set to eval mode
        self.model.eval()

        # Initialize graph (mentions are nodes and edges indicate coref linkage)
        graph = nx.Graph()

        # Pass the document through the model
        spans, probs = self.model(doc)

        # Cluster found coreference links
        for i, span in enumerate(spans):

            # Loss implicitly pushes coref links above 0, rest below 0
            found_corefs = [idx
                            for idx, _ in enumerate(span.yi_idx)
                            if probs[i, idx] > probs[i, len(span.yi_idx)]]

            # If we have any
            if any(found_corefs):

                # Add edges between all spans in the cluster
                for coref_idx in found_corefs:
                    link = spans[coref_idx]
                    graph.add_edge((span.i1, span.i2), (link.i1, link.i2))

        # Extract clusters as nodes that share an edge
        clusters = list(nx.connected_components(graph))

        # Initialize token tags
        token_tags = [[] for _ in range(len(doc))]

        # Add in cluster ids for each cluster of corefs in place of token tag
        for idx, cluster in enumerate(clusters):
            for i1, i2 in cluster:

                if i1 == i2:
                    token_tags[i1].append(f'({idx})')

                else:
                    token_tags[i1].append(f'({idx}')
                    token_tags[i2].append(f'{idx})')

        doc.tags = ['|'.join(t) if t else '-' for t in token_tags]

        return doc

    def to_conll(self, val_corpus):
        """ Write to out_file the predictions, return CoNLL metrics results """

        # Make predictions directory if there isn't one already
        golds_file, preds_file = 'preds/golds.txt', 'preds/predictions.txt'
        if not os.path.exists('preds/'):
            os.makedirs('preds/')

        # Combine all gold files into a single file (Perl script requires this)
        golds_file_content = flatten([doc.raw_text for doc in val_corpus])
        with io.open(golds_file, 'w', encoding='utf-8', errors='strict') as f:
            for line in golds_file_content:
                f.write(line)

        # Dump predictions
        with io.open(preds_file, 'w', encoding='utf-8', errors='strict') as f:

            for doc in val_corpus:

                current_idx = 0

                for line in doc.raw_text:

                    # Indicates start / end of document or line break
                    if line.startswith('#begin') or line.startswith('#end') or line == '\n':
                        f.write(line)
                        continue
                    else:
                        # Replace the coref column entry with the predicted tag
                        tokens = line.split()
                        tokens[-1] = doc.tags[current_idx]

                        # Increment by 1 so tags are still aligned
                        current_idx += 1

                        # Rewrite it back out
                        f.write('\t'.join(tokens))
                    f.write('\n')

        return golds_file, preds_file

    def save_model(self, path: Optional[str]=None):
        """ Save the model to a file """
        if path is None:
            path = os.path.join(CACHE, corefai.NAME[self.NAME])
        if not os.path.exists(path):
            os.makedirs(path)
        configs = dict(self.args)
        del configs['args']
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'state_dict': state_dict,
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict(),
                 'epoch': self.epoch,
                 }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(configs, f)
        torch.save(state, os.path.join(path, 'pytorch_model.pt'), pickle_module=dill)

    @classmethod
    def load_model(cls, path, src = 'gcp', reload = False, checkpoint=True,  **kwargs):
        """ Load state dictionary into model """
        args = Config(**locals())
        if not os.path.exists(path):
            path = download(src, corefai.MODEL[src].get(path, path), reload=reload)
        state = torch.load(os.path.join(path, 'pytorch_model.pt'), map_location='cpu')
        cls = corefai.RESOLVER[state['name']] if cls.NAME is None else cls
        optimizer = state['optimizer']
        scheduler = state['scheduler']
        epoch = state['epoch']
        
        #load configs
        with open(os.path.join(path, 'config.json'), 'r') as f:
            configs = json.load(f)
        args.update(configs)
        args.update({'checkpoint': checkpoint})

        resolver = cls(args)
        resolver.model.load_state_dict(state['state_dict'], False)
        resolver.optimizer.load_state_dict(optimizer)
        resolver.scheduler.load_state_dict(scheduler)
        resolver.epoch = epoch
        resolver.model.to(resolver.device)

        return resolver