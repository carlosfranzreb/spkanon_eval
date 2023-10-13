import torch
from omegaconf import OmegaConf

from .wavlm_model import WavLM, WavLMConfig


class WavlmWrapper:
    def __init__(self, config: OmegaConf, device: str, **kwargs):
        """
        Load the model from the checkpoint and move it to the device.

        Args:
            config: the configuration object. should include the path to the ckpt
            as `ckpt_path` and the layer from which to extract the features as
            `layer`.
            device: the device where the model should run.
        """
        ckpt = torch.load(config.ckpt_path, map_location=device)
        model_cfg = WavLM(ckpt["cfg"])
        self.model = WavLM(model_cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device
        self.config = config

    def run(self, batch: list) -> torch.Tensor:
        """
        Pases the batch through the model until the specified layer and returns the
        output of that layer.

        Args:
            batch: a list with a tensor comprising waveforms in first position.

        Returns:
            the output of the specified layer.
        """
        return self.model.extract_features(batch[0], output_layer=self.config.layer)[1]
