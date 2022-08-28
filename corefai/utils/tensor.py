import torch

def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def lookup_tensor(tokens, vectorizer):
    """ Convert a sentence to an embedding lookup tensor """
    return to_cuda(torch.tensor([vectorizer.stoi(t) for t in tokens]))

def speaker_label(s1, s2):
    """ Compute if two spans have the same speaker or not """
    # Same speaker
    if s1.speaker == s2.speaker:
        idx = torch.tensor(1)

    # Different speakers
    elif s1.speaker != s2.speaker:
        idx = torch.tensor(2)

    # No speaker
    else:
        idx = torch.tensor(0)

    return to_cuda(idx)