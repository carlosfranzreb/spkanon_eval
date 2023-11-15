from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from omegaconf import DictConfig
from torch import Tensor
from spkanon_eval.abstract_component import Component


class Wav2Vec(Component):
    def __init__(self, config: DictConfig, device: str, **kwargs) -> None:
        self.model = HuggingFaceWav2Vec2(
            config["hf_hub"],
            config["save_path"],
        )

    def run(self, batch: list) -> Tensor:
        return self.model(batch[0])
