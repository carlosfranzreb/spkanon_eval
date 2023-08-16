import torch


def pad_zeros_dim2(first, second):
    """Given two tensors, pad the smaller one with zeros so their shapes
    match on dimension 2. Their shapes can only differ in dimension 2!"""
    if first.shape[2] < second.shape[2]:
        padded = torch.zeros_like(second)
        padded[:, :, : first.shape[2]] = first
        return padded, second
    elif first.shape[2] > second.shape[2]:
        padded = torch.zeros_like(first)
        padded[:, :, : second.shape[2]] = second
        return first, padded
    else:
        return first, second
