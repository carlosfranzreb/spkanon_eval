"""
Wrapper for the Parallel WaveGAN model (https://pypi.org/project/parallel-wavegan/)
"""


from parallel_wavegan.utils import load_model
import torch


class ParallelWaveGAN:
    def __init__(self, config, device):
        self.device = device
        self.config = config
        self.model = load_model(config.init).to(device)
        self.model.remove_weight_norm()
        self.model.eval()
        self.replication_pad = torch.nn.ReplicationPad1d(self.model.aux_context_window)

    def run(self, batch):
        """
        Run the model on the given batch.
        Input dims: (batch_size, n_mels, n_frames)
        Output dims: (batch_size, n_samples)
        """
        spec = batch[self.config.input.spectrogram]
        x = torch.randn(
            spec.shape[0], 1, spec.shape[2] * self.model.upsample_factor
        ).to(self.device)
        c = self.replication_pad(spec)
        out = self.model.forward(x, c)
        return out

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(device)
