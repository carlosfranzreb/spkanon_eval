import importlib

from nemo.collections.tts.models import FastPitchModel
import torch
import torch.nn.functional as F


class FastPitch:
    def __init__(self, config, device):
        """
        The config must indicate under which key are placed the transcripts in the
        batch, under `config.input`. It may also indicate which speaker to use, under
        `config.speaker`. The model is loaded from the path specified in `config.init`.
        """

        self.device = device
        self.input_text = config.input.text
        self.input_source = config.input.source
        self.input_target = config.input.target
        self.n_targets = config.n_targets

        model_path = config.init
        if model_path.endswith(".nemo"):
            self.model = FastPitchModel.restore_from(restore_path=model_path)
        elif model_path.endswith(".ckpt"):
            self.model = FastPitchModel.load_from_checkpoint(checkpoint_path=model_path)
        else:
            self.model = FastPitchModel.from_pretrained(model_name=model_path)
        self.model.eval()
        self.target_selection = None  # initialized later (see init_target_selection)

    def init_target_selection(self, cfg, *args):
        """
        Initialize the target selection algorithm. This method is called by the
        anonymizer, passing it config and the arguments that the defined algorithm
        requires. These are passed directly to the algorithm, along with the style
        vectors of the StarGAN.
        """

        targets = torch.arange(self.n_targets).to(self.device)
        module_str, cls_str = cfg.cls.rsplit(".", 1)
        module = importlib.import_module(module_str)
        cls = getattr(module, cls_str)
        self.target_selection = cls(targets, cfg, *args)

    def run(self, batch):
        """
        The input `batch` is a dict with a key `self.input` that contains the
        transcripts. For each transcript, parse it with the model's parser (performs
        text normalization and character tokenization) and then generate a spectrogram
        with those tokens. Return the spectrograms as a tensor.
        """

        # get the texts and the source and target speakers
        texts = batch[self.input_text]
        source = batch[self.input_source]
        target_in = batch[self.input_target] if self.input_target in batch else None
        mock_input = torch.zeros(len(texts), dtype=torch.int64, device=self.device)
        target = self.target_selection.select(mock_input, source, target_in)

        # compute the tokens from the transcripts
        tokens = list()
        for text in texts:
            tokens.append(self.model.parse(text).squeeze())
        tokens = _pad_tensors(tokens)

        spec = self.model.generate_spectrogram(tokens, speaker=target)
        return {"spectrogram": spec, "target": target}

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.model.to(device)
        self.device = device


def _pad_tensors(lists):
    """`lists` is a list of tensors. Pad them so they have the same length and
    make a tensor with them."""
    max_values = max([l.shape[0] for l in lists])
    return torch.cat(
        [F.pad(l, (0, max_values - l.shape[0]), value=0).unsqueeze(0) for l in lists]
    )
