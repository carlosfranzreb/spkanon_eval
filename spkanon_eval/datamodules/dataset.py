"""
Create a dataset from a list of datafiles. The datafiles should be text files, each
comprising one JSON object per line. Each JSON object should have at least the
following keys:

- "audio_filepath": path to the audio file
- "label": speaker ID

The dataset returns tuples of (audio, speaker), where the audio is resampled to the
given sampling rate.
"""

import json
import os
import logging

import torch
import torchaudio


LOGGER = logging.getLogger("progress")


class SpeakerIdDataset(torch.utils.data.Dataset):
    def __init__(self, datafiles: list[str], sample_rate: int) -> None:
        """
        - Store the number of samples in each datafile.
        - Store a mapping from speaker IDs to indices.
        - Dump the mapping to the experiment folder.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.datafiles = datafiles

        self.n_samples = list()
        self.spk2id = dict()
        for datafile in self.datafiles:
            df_base = os.path.splitext(os.path.basename(datafile))[0]
            self.n_samples.append(0)
            with open(datafile) as f:
                for line in f:
                    obj = json.loads(line)
                    spk = df_base + "-" + obj["label"]
                    if spk not in self.spk2id:
                        self.spk2id[spk] = len(self.spk2id)
                    self.n_samples[-1] += 1

        # dump the mapping to the experiment folder
        datafile_dir = os.path.dirname(datafiles[0])
        log_dir = datafile_dir[: datafile_dir.rfind("/data")]
        with open(os.path.join(log_dir, "spk2id.json"), "w") as f:
            json.dump(self.spk2id, f)

        LOGGER.info(f"Dataset created for datafiles `{datafiles}`")
        LOGGER.info(f"Number of samples: {sum(self.n_samples)}")
        LOGGER.info(f"Number of speakers: {len(self.spk2id)}")

    def __len__(self) -> int:
        """Return the number of samples in all datafiles."""
        return sum(self.n_samples)

    def __getitem__(self, sample_idx: int) -> tuple[torch.Tensor, str]:
        """
        Return the `sample_idx`-th sample of the dataset. The id is the index of the
        sample when considering all the datafiles as a single dataset. The item is
        returned as a tuple of (audio, speaker), where the audio is resampled to the
        given sampling rate.
        """

        # find the datafile containing the id-th sample
        for datafile_idx, n_samples in enumerate(self.n_samples):
            if sample_idx < n_samples:
                break
            sample_idx -= n_samples

        # load the audio and the tokens
        with open(self.datafiles[datafile_idx]) as f:
            for line_idx, line in enumerate(f):
                if line_idx == sample_idx:
                    break

        obj = json.loads(line)
        audio = load_audio(obj["audio_filepath"], self.sample_rate)
        df_base = os.path.splitext(os.path.basename(self.datafiles[datafile_idx]))[0]
        spk = df_base + "-" + obj["label"]
        return (audio, self.spk2id[spk])


def load_audio(audio_path: str, sample_rate: int) -> torch.Tensor:
    """
    Load the audio from the given path. If the sampling rate is different from
    given sampling rate, resample the audio. Return the waveform as a 1D tensor.
    If the audio is stereo, return only the first channel.
    """

    audio, sr = torchaudio.load(audio_path, normalize=True)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)

    if audio.ndim > 1:
        return audio[0]
    else:
        return audio.squeeze(0)
