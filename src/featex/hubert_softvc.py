"""
HuBERT component of the SoftVC model.
"""

import torch


SAMPLE_RATE = 16000  # model's sample rate


class HubertSoftVC:
    def __init__(self, config, device):
        """
        The model is loaded following the instructions in the notebook provided in
        the repository https://github.com/bshall/soft-vc.
        """
        self.device = device
        self.model = torch.hub.load(
            "bshall/hubert:main", "hubert_soft", force_reload=True
        )
        self.model.to(self.device)
        self.model.eval()

    def run(self, batch):
        """
        Returns the acoustic units for the given NeMo batch, which is a tuple where
        the audio batch is placed in the first position.
        """
        with torch.inference_mode():
            return self.model.units(batch[0].unsqueeze(1))

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
