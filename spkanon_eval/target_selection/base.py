import logging
import torch
from torch import Tensor
from omegaconf import OmegaConf


LOGGER = logging.getLogger("progress")


class BaseSelector:
    def __init__(self, vecs: list, cfg: OmegaConf) -> None:
        """
        Initialize the target selector with the style vectors and a flag indicating
        whether the targets should be consistent across the utterances. If this flag
        is set to True, the targets are stored in a dictionary. This class takes care
        of updating the dictionary with the new targets and decides when a new target
        has to be selected. Selectors that inherit this class must implement the
        `select_new` method, where a new target is selected for a given utterance.
        """
        self.vecs = vecs
        self.targets = dict() if cfg.consistent_targets else None

    def select(self, spec: Tensor, source: Tensor, target: Tensor = None) -> Tensor:
        """
        Targets may be -1, in which case the target selection algorithm is used. If
        speaker consistency is enabled, the target speakers must be consistent across
        the utterances of each speaker. This is checked here and may overwrite the
        given target speaker.
        """
        # create the target tensor if it is not given
        if target is None:
            target = (
                torch.ones(spec.shape[0], dtype=torch.int64, device=spec.device) * -1
            )

        # if speaker consistency is disabled, compute undefined targets and return them
        if self.targets is None:
            mask = target == -1
            target[mask] = self.select_new(spec[mask])
            return target

        # overwrite current targets
        for idx, src in enumerate(source):
            src = src.item()
            if src in self.targets and self.targets[src] != target[idx]:
                target[idx] = self.targets[src]

        # if a target is already defined, ensure that it is consistent across sources
        for idx in torch.argwhere(target != -1):
            if target[idx] == -1:
                continue
            for j in range(target.shape[0]):
                if idx == j:
                    continue
                if source[idx] == source[j] and target[idx] != target[j]:
                    target[j] = target[idx]

        # compute the target for each unique source that is not already defined
        new_sources = list()
        new_source_indices = list()
        for idx, src in enumerate(source):
            if target[idx] == -1 and src not in new_sources:
                new_sources.append(src.item())
                new_source_indices.append(idx)
        if len(new_sources) > 0:
            new_targets = self.select_new(spec[new_source_indices])

        # update the output targets and the stored targets
        for idx, src in enumerate(source):
            if target[idx] == -1:
                target[idx] = new_targets[new_sources.index(src)]
            if src not in self.targets:
                self.targets[src.item()] = target[idx].item()

        return target

    def select_new(self, spec: Tensor) -> Tensor:
        """
        Select a new target speaker style vector for the given input spectrogram.
        """
        raise NotImplementedError

    def set_consistent_targets(self, consistent_targets: bool) -> None:
        """
        Update the target selection algorithm with the new value of
        `consistent_targets`.
        """
        if consistent_targets is True:
            LOGGER.info("Enabling consistent targets and removing previous targets")
            self.targets = dict()
        elif consistent_targets is False:
            LOGGER.info("Disabling consistent targets")
            self.targets = None
