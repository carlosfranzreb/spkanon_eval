"""
- Wrapper for speaker recognition models. The `run` method returns speaker embeddings.
- It may use pre-trained models from both SpeechBrain and NeMo.
"""


import os
import logging

from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from speechbrain.pretrained import EncoderClassifier


LOGGER = logging.getLogger("progress")
SAMPLE_RATE = 16000


class SpkId:
    def __init__(self, config, device):
        """Initialize the model with the given config and freeze its parameters."""
        self.config = config
        self.device = device
        self.toolkit = config.toolkit
        model_path = config.path

        if self.toolkit == "nemo":
            if model_path.endswith(".nemo"):
                self.model = EncDecSpeakerLabelModel.restore_from(
                    restore_path=model_path
                )
            elif model_path.endswith(".ckpt"):
                self.model = EncDecSpeakerLabelModel.load_from_checkpoint(
                    checkpoint_path=model_path
                )
            else:
                self.model = EncDecSpeakerLabelModel.from_pretrained(
                    model_name=model_path
                )
            self.model.to(device)

        elif self.toolkit == "speechbrain":
            self.model = EncoderClassifier.from_hparams(
                source=model_path,
                savedir=os.path.join("checkpoints", model_path),
                run_opts={"device": device},
            )

        else:
            raise ValueError(f"Unknown toolkit for spkid: {config.toolkit}")

        self.model.eval()

    def run(self, batch):
        if self.toolkit == "nemo":
            return self.model.forward(
                input_signal=batch[0].to(self.device),
                input_signal_length=batch[1].to(self.device),
            )[1]
        elif self.toolkit == "speechbrain":
            return self.model.encode_batch(batch[0].to(self.device)).squeeze(1)

    def finetune(self, exp_folder, datafiles, n_speakers):
        raise NotImplementedError
