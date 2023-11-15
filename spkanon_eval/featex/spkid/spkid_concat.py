"""
Concatenates the output of several spkid models. It receives a list of spkid models,
which it runs in sequence, and concatenates their outputs. Each spkid model is a
spkanon_eval.featex.spkid.spkid.SpkId object.
"""


import logging

import torch

from spkanon_eval.featex.spkid.spkid import SpkId
from spkanon_eval.component_definitions import InferComponent


LOGGER = logging.getLogger("progress")
SAMPLE_RATE = 16000


class SpkIdConcat(InferComponent):
    def __init__(self, config, device):
        """Initialize the model with the given config and freeze its parameters."""
        self.config = config
        self.device = device
        self.models = [SpkId(cfg, device) for cfg in config.models]

    def run(self, batch):
        spkembs = [model.run(batch) for model in self.models]
        return torch.cat(spkembs, dim=1)

    def to(self, device: str) -> None:
        for model in self.models:
            model.to(device)
        self.device = device
