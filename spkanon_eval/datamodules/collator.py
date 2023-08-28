from torch import tensor
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: list[tuple[tensor, int]]) -> tuple[tensor, tensor]:
    """
    Collates tuples of (audio, speaker) into batches of audio and speakers. The audio
    are 1D tensors, and the speaker IDs are integers.
    """
    audio, speakers = zip(*batch)
    audio = pad_sequence(audio, batch_first=True)
    speakers = tensor([int(speaker) for speaker in speakers])
    return audio, tensor(speakers)
