"""
Acoustic model for the SoftVC model. It is preceded by a HuBERT model and followed
by a HiFiGAN model.
"""


import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from submodules.softvc_acoustic.acoustic.model import AcousticModel


SAMPLE_RATE = 16000  # model's sample rate


class AcousticSoftVC:
    def __init__(self, config, device):
        """
        - The config must indicate under which key are placed the transcripts in the
            batch, under `config.input`..
        - The model is loaded following the instructions in the notebook provided in
            the repository https://github.com/bshall/soft-vc.
        """
        self.device = device
        self.input = config.input.hubert_units

        self.model = AcousticModel(False, True)
        acoustic_ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/bshall/acoustic-model/releases/download/v0.1/hubert-soft-0321fd7e.pt",
            map_location="cpu",
        )
        consume_prefix_in_state_dict_if_present(
            acoustic_ckpt["acoustic-model"], "module."
        )
        self.model.load_state_dict(acoustic_ckpt["acoustic-model"])
        self.model.to(device)
        self.model.eval()

    def run(self, batch):
        """
        Given the HuBERT units, placed in the batch under the key `self.input`,
        computes and returns the spectrogram.
        """
        with torch.inference_mode():
            spec = self.model.generate(batch[self.input]).transpose(1, 2)
            return {"spectrogram": spec, "target": 0}

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
