import torch

from spkanon_eval.target_selection.base import BaseSelector


class RandomSelector(BaseSelector):
    def select_new(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Randomly select a target for the given input spectrogram.
        """
        return torch.randint(0, len(self.vecs), (spec.shape[0],), device=spec.device)
