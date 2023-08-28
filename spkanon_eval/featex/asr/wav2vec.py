from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2


class Wav2Vec:
    def __init__(self, config):
        self.model = HuggingFaceWav2Vec2(
            config["hf_hub"],
            config["save_path"],
        )

    def run(self, batch):
        return self.model(batch[0])
