from torch import tensor
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: list[tuple[tensor, int, int]]) -> tuple[tensor, tensor, tensor]:
    """
    Collates tuples of (audio, speaker) into batches of waveforms, speakers and
    lengths. The waveforms are 1D tensors, and the speaker IDs and waveform lengths
    are integers.
    """
    audio, speakers, n_samples = zip(*batch)
    audio = pad_sequence(audio, batch_first=True)
    speakers = tensor([int(speaker) for speaker in speakers])
    n_samples = tensor(n_samples)
    return audio, speakers, n_samples
