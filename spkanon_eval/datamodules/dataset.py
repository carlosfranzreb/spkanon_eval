"""
Create a dataset from a list of datafiles. The datafiles should be text files, each
comprising one JSON object per line. The dataset returns tuples of (audio, speaker),
where the audio is resampled to the given sampling rate.
"""

import json
import logging

import torch
import torchaudio


LOGGER = logging.getLogger("progress")


class SpeakerIdDataset(torch.utils.data.Dataset):
    def __init__(self, datafile: str, sample_rate: int) -> None:
        """
        Create a dataset from the given datafile. The datafile should be a text file,
        comprising one JSON object per line. Each JSON object should have at least the
        following keys:

        - "path": path to the audio file
        - "speaker_id": a unique integer identifying the speaker
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.datafile = datafile
        self.n_samples = 0
        with open(self.datafile) as f:
            for _ in f:
                self.n_samples += 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, sample_idx: int) -> tuple[torch.Tensor, str]:
        """
        Return the `sample_idx`-th sample of the dataset. The id is the index of the
        sample when considering all the datafiles as a single dataset. The item is
        returned as a tuple of (audio, speaker, n_samples), where the audio is
        resampled to the given sampling rate.
        """

        # load the audio and the tokens
        with open(self.datafile) as f:
            for line_idx, line in enumerate(f):
                if line_idx == sample_idx:
                    break

        obj = json.loads(line)
        audio = load_audio(obj["path"], self.sample_rate)
        return (audio, obj["speaker_id"], audio.shape[0])


def load_audio(audio_path: str, sample_rate: int) -> torch.Tensor:
    """
    Load the audio from the given path. If the sampling rate is different from
    given sampling rate, resample the audio. Return the waveform as a 1D tensor.
    If the audio is stereo, returns the mean across channels.
    """

    audio, sr = torchaudio.load(audio_path, normalize=True)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    audio = audio.squeeze()
    if audio.ndim > 1:
        audio = audio.mean(dim=0)
    return audio
