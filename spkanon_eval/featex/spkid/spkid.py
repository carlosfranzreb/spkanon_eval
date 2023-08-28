"""
Wrapper for Speechbrain speaker recognition models. The `run` method returns speaker
embeddings.
"""


import os
import logging

from speechbrain.pretrained import EncoderClassifier


LOGGER = logging.getLogger("progress")
SAMPLE_RATE = 16000


class SpkId:
    def __init__(self, config, device):
        """Initialize the model with the given config and freeze its parameters."""
        self.config = config
        self.device = device
        model_path = config.path
        self.model = EncoderClassifier.from_hparams(
            source=model_path,
            savedir=os.path.join("checkpoints", model_path),
            run_opts={"device": device},
        )
        self.model.eval()

    def run(self, batch):
        return self.model.encode_batch(batch[0].to(self.device)).squeeze(1)

    def finetune(self, exp_folder, datafiles, n_speakers):
        return NotImplementedError
