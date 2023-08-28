"""
Returns the same spectrograms it receives. Used for testing purposes.
"""


import importlib
import torch


class DummyConverter:
    def __init__(self, config, device):
        """
        Store where the spectrogram, source and target is stored in the batch.
        """
        self.input_spec = config.input.spectrogram
        self.input_source = config.input.source
        self.input_target = config.input.target
        self.device = device
        self.model = torch.empty(1)
        self.n_targets = config.n_targets
        self.target_selection = None

    def run(self, batch):
        """
        Return the given spectrograms and the targets.
        """
        spec = batch[self.input_spec]
        source = batch[self.input_source]
        target_in = batch[self.input_target] if self.input_target in batch else None
        mock_input = torch.zeros(spec.shape[0], dtype=torch.int64, device=self.device)
        target = self.target_selection.select(mock_input, source, target_in)
        return {"spectrogram": spec, "target": target}

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(device)

    def init_target_selection(self, cfg, *args):
        """
        Initialize the target selection.
        """

        targets = torch.arange(self.n_targets).to(self.device)
        module_str, cls_str = cfg.cls.rsplit(".", 1)
        module = importlib.import_module(module_str)
        cls = getattr(module, cls_str)
        self.target_selection = cls(targets, cfg, *args)
