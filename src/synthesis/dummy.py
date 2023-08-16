"""
Synthesizer that does nothing, to be used in testing.
"""


import torch


class DummySynthesizer:
    def __init__(self, config, device):
        """
        Store where the spectrogram is stored in the batch.
        """
        self.spec_input = config.input.spectrogram
        self.device = device
        self.model = torch.empty(1)

    def run(self, batch):
        """
        Return a tensor of random values of shape (batch_size, 1, spec_frames * 512).
        """
        specs = batch[self.spec_input]
        length = specs.shape[2] * 512
        return torch.randn((specs.shape[0], 1, length), device=self.device)

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(device)
