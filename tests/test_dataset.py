import unittest
import os
import json
import logging
import shutil

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
        self.log_dir = "tests/logs/test_dataset"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "data"), exist_ok=True)

        datafile_dir = "tests/datafiles"
        self.datafiles = list()
        for f in os.listdir("tests/datafiles"):
            target = os.path.join(self.log_dir, "data", f)
            shutil.copy(os.path.join(datafile_dir, f), target)
            self.datafiles.append(target)

        log_file = "progress"
        self.logger = logging.getLogger(log_file)
        self.logger.addHandler(
            logging.FileHandler(os.path.join(self.log_dir, f"{log_file}.log"))
        )
        self.sample_rate = 16000
        self.dataset = SpeakerIdDataset(self.datafiles, self.sample_rate)

    def tearDown(self) -> None:
        """Delete the log directory and its contents."""
        shutil.rmtree(self.log_dir)

    def test_dataset_length(self):
        n_samples = 0
        for datafile in self.datafiles:
            with open(datafile) as f:
                n_samples += len(f.readlines())
        self.assertEqual(len(self.dataset), n_samples)

    def test_spk2id(self):
        """
        The dataset maps speaker Ids to integers. The mapping is stored in a JSON
        file in the experiment folder. Check that this file exists and that the
        mapping comprises the correct speakers.
        """
        spk2id_path = os.path.join(self.log_dir, "spk2id.json")
        self.assertTrue(os.path.exists(spk2id_path))
        spk2id = json.load(open(spk2id_path))

        unique_speakers = list()
        for datafile in self.datafiles:
            df_base = os.path.splitext(os.path.basename(datafile))[0]
            for line in open(datafile):
                obj = json.loads(line)
                spk = df_base + "-" + obj["label"]
                self.assertTrue(spk in spk2id)
                if spk not in unique_speakers:
                    unique_speakers.append(spk)
        self.assertEqual(len(spk2id), len(unique_speakers))

    def test_dataset_items(self):
        """
        Ensure that the dataset returns the correct audio and speaker IDs. We assume
        that the dataset's spk2id mapping is correct.
        """

        self.assertTrue(hasattr(self.dataset, "spk2id"))
        sample_idx = 0
        for datafile in self.datafiles:
            df_base = os.path.splitext(os.path.basename(datafile))[0]
            for line in open(datafile).readlines():
                obj = json.loads(line)
                audio_true, sr = torchaudio.load(obj["audio_filepath"])
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.sample_rate
                    )
                    audio_true = resampler(audio_true)
                audio_dataset, spk = self.dataset[sample_idx]
                spk_true = self.dataset.spk2id[df_base + "-" + obj["label"]]
                self.assertEqual(audio_true.shape[1], audio_dataset.shape[0])
                self.assertEqual(spk_true, spk)
                self.assertTrue(torch.allclose(audio_true, audio_dataset))
                sample_idx += 1
