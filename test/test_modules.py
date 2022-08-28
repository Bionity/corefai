from corefai.modules.ffnn import Score, Distance, Genre, Speaker
from corefai.modules.cnn import CharCNN

import torch

def test_score():
    emb = torch.rand(10, 100)
    scores = Score(embeds_dim=100, hidden_dim=10)
    score = scores(emb)
    assert score.shape == torch.Size([10, 1])

def test_distance():
    dist = Distance(distance_dim=10)
    dist = dist(torch.tensor([1,2,3,4,5,6,7,8,9,10]))
    assert dist.shape == torch.Size([10, 10])

def test_genre():
    genre = Genre(genre_dim=10)
    genre = genre(['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb'])
    assert genre.shape == torch.Size([7, 10])

def test_speaker():
    speaker = Speaker(speaker_dim=10)
    speaker = speaker([0, 1, 1, 1, 0])
    assert speaker.shape == torch.Size([5, 10])

def test_char_cnn():
    charcnn = CharCNN(vocab=set('abcdefghijklmnopqrstuvwxyz'), filters=10, char_dim=8)
    pooled = charcnn(['a', 'b', 'c'])
    assert pooled.shape == (3, 10)