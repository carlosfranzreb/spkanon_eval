import logging
import torch


LOGGER = logging.getLogger("progress")
TARGET_LOGGER = logging.getLogger("targets")


class BaseSelector:
    def __init__(self, vecs, cfg):
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

    def select(self, spec, source, target=None):
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
            self.log_targets(source, target)
            return target

        # overwrite current targets
        for i, s in enumerate(source):
            if s in self.targets and self.targets[s] != target[i]:
                target[i] = self.targets[s]

        # if a target is already defined, ensure that it is consistent across sources
        for i in torch.argwhere(target != -1):
            if target[i] == -1:
                continue
            for j in range(target.shape[0]):
                if i == j:
                    continue
                if source[i] == source[j] and target[i] != target[j]:
                    target[j] = target[i]

        # compute the target for each unique source that is not already defined
        unique_source = list()
        for i in range(len(source)):
            if i not in unique_source and target[i] == -1:
                unique_source.append(i)

        if len(unique_source) > 0:
            unique_target = self.select_new(spec[unique_source])

        # update the output targets and the stored targets
        for i, s in enumerate(source):
            if target[i] == -1:
                target[i] = unique_target[unique_source.index(source.index(s))]
            if s not in self.targets:
                self.targets[s] = target[i]

        self.log_targets(source, target)
        return target

    def select_new(self, spec):
        """
        Select a new target speaker style vector for the given input spectrogram.
        """
        raise NotImplementedError

    def set_consistent_targets(self, consistent_targets):
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

    def log_targets(self, source, target):
        """
        Log each source-target pair.
        """
        for s, t in zip(source, target):
            TARGET_LOGGER.info(f"{s} -> {t}")
