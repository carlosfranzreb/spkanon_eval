import unittest
import json

import torch
import torchaudio

from spkanon_eval.datamodules.dataset import SpeakerIdDataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        """
        - Create a test directory and add a logger there; the spk2id mapping will be
            dumped there.
        - Get the test datafiles and create the dataset.
        """
        self.datafile = "spkanon_eval/tests/datafiles/ls-dev-clean-2.txt"
        self.sample_rate = 16000
        self.dataset = SpeakerIdDataset(self.datafile, self.sample_rate)

    def test_dataset_length(self):
        n_samples = 0
        with open(self.datafile) as f:
            n_samples += len(f.readlines())
        self.assertEqual(len(self.dataset), n_samples)

    def test_dataset_items(self):
        """
        Ensure that the dataset returns the correct audio, speaker IDs and audio
        durations. We assume that the dataset's spk2id mapping is correct.
        """

        sample_idx = 0
        for line in open(self.datafile).readlines():
            obj = json.loads(line)
            audio_true, sr = torchaudio.load(obj["path"])
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.sample_rate
                )
                audio_true = resampler(audio_true)
            audio_true = audio_true.squeeze()
            audio_dataset, spk, n_samples = self.dataset[sample_idx]
            self.assertEqual(audio_true.shape[0], audio_dataset.shape[0])
            self.assertEqual(obj["speaker_id"], spk)
            self.assertTrue(torch.allclose(audio_true, audio_dataset))
            self.assertEqual(n_samples, audio_true.shape[0])
            sample_idx += 1
