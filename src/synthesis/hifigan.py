from nemo.collections.tts.models import HifiGanModel


class HifiGan:
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        model_init = config.init
        if model_init.endswith(".yaml"):
            self.model = HifiGanModel(config)
        elif model_init.endswith(".nemo"):
            self.model = HifiGanModel.restore_from(restore_path=model_init)
        elif model_init.endswith(".ckpt"):
            self.model = HifiGanModel.load_from_checkpoint(checkpoint_path=model_init)
        else:
            self.model = HifiGanModel.from_pretrained(model_name=model_init)

    def run(self, batch):
        spec = batch[self.config.input.spectrogram]
        return self.model.forward(spec=spec)

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(device)
