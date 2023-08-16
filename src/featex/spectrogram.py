"""
Compute the spectrogram of the audio tensor after normalizing it.
"""

import logging
import torch
from torchaudio.transforms import MelSpectrogram

LOGGER = logging.getLogger("progress")

class SpecExtractor:
    def __init__(self, config, device):
        self.config = config
        self.mel_trafo = MelSpectrogram(
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
        ).to(device)

        LOGGER.info("Initialized mel-spectrogram extractor with config:")
        LOGGER.info(f"\tn_mels: {config.n_mels}")
        LOGGER.info(f"\tn_fft: {config.n_fft}")
        LOGGER.info(f"\twin_length: {config.win_length}")
        LOGGER.info(f"\thop_length: {config.hop_length}")

    def run(self, batch):
        audio_norm = batch[0] / torch.max(torch.abs(batch[0]))
        return self.mel_trafo(audio_norm)
    
    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.mel_trafo.to(device)
