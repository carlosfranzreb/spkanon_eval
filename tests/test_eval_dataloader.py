import unittest
import os
import json
import logging
import shutil

from omegaconf import OmegaConf
import torchaudio
import torch

from spkanon_eval.datamodules.dataloader import eval_dataloader


class TestEvalDataloader(unittest.TestCase):
    def setUp(self):
        """
        - Create a test directory and add a logger there; the spk2id mapping will be
            dumped there.
        - Get the test datafiles.
        """
        datafile_dir = "tests/datafiles"
        self.log_dir = "tests/logs/test_dataloader"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "data"), exist_ok=True)
        self.datafiles = list()
        for f in os.listdir(datafile_dir):
            target = os.path.join(self.log_dir, "data", f)
            shutil.copy(os.path.join(datafile_dir, f), target)
            self.datafiles.append(target)

        self.sample_rate = 16000
        self.batch_size = 2
        self.config = OmegaConf.create(
            {
                "trainer": {
                    "batch_size": self.batch_size,
                    "num_workers": 0,
                },
                "sample_rate": self.sample_rate,
            }
        )
        self.device = "cpu"

        os.makedirs(self.log_dir, exist_ok=True)
        log_file = "progress"
        self.logger = logging.getLogger(log_file)
        self.logger.addHandler(
            logging.FileHandler(os.path.join(self.log_dir, f"{log_file}.log"))
        )

    def tearDown(self) -> None:
        """Delete the log directory and its contents."""
        shutil.rmtree(self.log_dir)

    def test_eval_dataloader(self):
        """
        Test that the eval dataloader returns the correct number of batches and that
        the content of the batches matches that of the test datafiles. We asume that
        the dataset's spk2id mapping is correct.
        """
        dl = eval_dataloader(self.config, self.datafiles, self.device)
        samples = dict()
        for datafile in self.datafiles:
            samples[datafile] = open(datafile).readlines()

        for datafile, batch, data in dl:
            batch_size = batch[0].shape[0]
            self.assertLessEqual(batch_size, self.batch_size)
            objs = list()
            for _ in range(batch_size):
                objs.append(json.loads(samples[datafile].pop(0)))

            self.assertEqual(len(batch), 2)  # audio, speakers
            for i in range(batch_size):
                obj = objs[i]
                audio_true, sr = torchaudio.load(obj["audio_filepath"])
                sample_data = data[i]
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.sample_rate
                    )
                    audio_true = resampler(audio_true)
                self.assertTrue(audio_true.shape[1] <= batch[0][i].shape[0])
                self.assertTrue(
                    torch.allclose(audio_true, batch[0][i, : audio_true.shape[1]])
                )
                self.assertTrue(torch.sum(batch[0][i, audio_true.shape[1] :]) == 0)

                # check metadata
                for key in obj.keys():
                    if key == "audio_filepath":
                        continue
                    a, b = obj[key], sample_data[key]
                    if not isinstance(a, str):
                        a = str(a)
                    if not isinstance(b, str):
                        b = str(b)
                    self.assertEqual(a, b)

        # ensure that all samples have been read
        for datafile in self.datafiles:
            self.assertEqual(len(samples[datafile]), 0)
