import torch
from torch import nn
from torch.nn import functional as F

import attr
from typing import List

from corefai.structs.dataset import Span, Document
from corefai.modules.ffnn import Score, Distance, Genre, Speaker
from corefai.utils.transforms import compute_idx_spans, \
                                        pad_and_stack, prune, pairwise_indexes
from corefai.utils.tensor import speaker_label, to_var, to_cuda

class MentionScore(nn.Module):
    """ Mention scoring module
    """
    def __init__(self, gi_dim: int, attn_dim: int, distance_dim: int):
        """ Initialize the mention scoring module
            Args:
                gi_dim (int): dimension of the span representation
                attn_dim (int): dimension of the attention representation
                distance_dim (int): dimension of the distance representation
        """
        super().__init__()

        self.attention = Score(attn_dim)
        self.width = Distance(distance_dim)
        self.score = Score(gi_dim)

    def forward(self, states: torch.Tensor, embeds: torch.Tensor, doc: Document, K=250):
        """ Compute unary mention score for each span
            Args:
                states (List[Tensor]): list of hidden states for each span
                embeds (Tensor): input embeddings for each sentence
                doc (Document): document object
                K (int): number of top scoring spans to return
        """

        # Initialize Span objects containing start index, end index, genre, speaker
        spans = [Span(i1=i[0], i2=i[-1], id=idx,
                      speaker=doc.speaker(i), genre=doc.genre)
                 for idx, i in enumerate(compute_idx_spans(doc.sents))]

        # Compute first part of attention over span states (alpha_t)
        attns = self.attention(states)

        # Regroup attn values, embeds into span representations
        # TODO: figure out a way to batch
        span_attns, span_embeds = zip(*[(attns[s.i1:s.i2+1], embeds[s.i1:s.i2+1])
                                        for s in spans])

        # Pad and stack span attention values, span embeddings for batching
        padded_attns, _ = pad_and_stack(span_attns, value=-1e10)
        padded_embeds, _ = pad_and_stack(span_embeds)

        # Weight attention values using softmax
        attn_weights = F.softmax(padded_attns, dim=1)

        # Compute self-attention over embeddings (x_hat)
        attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)

        # Compute span widths (i.e. lengths), embed them
        widths = self.width([len(s) for s in spans])

        # Get LSTM state for start, end indexes
        # TODO: figure out a way to batch
        start_end = torch.stack([torch.cat((states[s.i1], states[s.i2]))
                                 for s in spans])

        # Cat it all together to get g_i, our span representation
        g_i = torch.cat((start_end, attn_embeds, widths), dim=1)

        # Compute each span's unary mention score
        mention_scores = self.score(g_i)

        # Update span object attributes
        # (use detach so we don't get crazy gradients by splitting the tensors)
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores.detach())
        ]

        # Prune down to LAMBDA*len(doc) spans
        spans = prune(spans, len(doc))

        # Update antencedent set (yi) for each mention up to K previous antecedents
        spans = [
            attr.evolve(span, yi=spans[max(0, idx-K):idx])
            for idx, span in enumerate(spans)
        ]

        return spans, g_i, mention_scores


class PairwiseScore(nn.Module):
    """ Coreference pair scoring module
    """
    def __init__(self, gij_dim: int, distance_dim: int, genre_dim: int, speaker_dim: int):
        """ Initialize the pairwise scoring module
            Args:
                gij_dim (int): dimension of the span representation
                distance_dim (int): dimension of the distance representation
                genre_dim (int): dimension of the genre representation
                speaker_dim (int): dimension of the speaker representation
        """
        super().__init__()

        self.distance = Distance(distance_dim)
        self.genre = Genre(genre_dim)
        self.speaker = Speaker(speaker_dim)

        self.score = Score(gij_dim)

    def forward(self, spans: List[Span], g_i: torch.Tensor, mention_scores: torch.Tensor):
        """ Compute pairwise score for spans and their up to K antecedents
            Args:
                spans (List[Span]): list of Span objects
                g_i (Tensor): span representation for each span
                mention_scores (Tensor): unary mention score for each span
        """
        # Extract raw features


        mention_ids, antecedent_ids, \
            distances, genres, speakers = zip(*[(i.id, j.id,
                                                i.i2-j.i1, i.genre,
                                                speaker_label(i, j))
                                             for i in spans
                                             for j in i.yi])

        # For indexing a tensor efficiently
        mention_ids = to_cuda(torch.tensor(mention_ids))
        antecedent_ids = to_cuda(torch.tensor(antecedent_ids))

        # Embed them
        phi = torch.cat((self.distance(distances),
                         self.genre(genres),
                         self.speaker(speakers)), dim=1)

        # Extract their span representations from the g_i matrix
        i_g = torch.index_select(g_i, 0, mention_ids)
        j_g = torch.index_select(g_i, 0, antecedent_ids)

        # Create s_ij representations
        pairs = torch.cat((i_g, j_g, i_g*j_g, phi), dim=1)

        # Extract mention score for each mention and its antecedents
        s_i = torch.index_select(mention_scores, 0, mention_ids)
        s_j = torch.index_select(mention_scores, 0, antecedent_ids)

        # Score pairs of spans for coreference link
        s_ij = self.score(pairs)

        # Compute pairwise scores for coreference links between each mention and
        # its antecedents
        coref_scores = torch.sum(torch.cat((s_i, s_j, s_ij), dim=1), dim=1, keepdim=True)

        # Update spans with set of possible antecedents' indices, scores
        spans = [
            attr.evolve(span,
                        yi_idx=[((y.i1, y.i2), (span.i1, span.i2)) for y in span.yi]
                        )
            for span, score, (i1, i2) in zip(spans, coref_scores, pairwise_indexes(spans))
        ]

        # Get antecedent indexes for each span
        antecedent_idx = [len(s.yi) for s in spans if len(s.yi)]
        if antecedent_idx == []:
            antecedent_idx = [1]
        # Split coref scores so each list entry are scores for its antecedents, only.
        # (NOTE that first index is a special case for torch.split, so we handle it here)
        split_scores = [to_cuda(torch.tensor([]))] \
                         + list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = to_var(torch.tensor([[0.]]))
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        # Batch and softmax
        # get the softmax of the scores for each span in the document given
        probs = [F.softmax(tensr,dim=0) for tensr in with_epsilon]
        
        # pad the scores for each one with a dummy value, 1000 so that the tensors can 
        # be of the same dimension for calculation loss and what not. 
        probs, _ = pad_and_stack(probs, value=1000)
        probs = probs.squeeze()
       
        return spans, probs
