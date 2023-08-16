import torch

from src.target_selection.base import BaseSelector


class RandomSelector(BaseSelector):
    def select_new(self, spec):
        """
        Randomly select a target for the given input spectrogram.
        """
        return torch.randint(
            0, self.vecs.shape[0], (spec.shape[0],), device=spec.device
        )
