"""
Test that the spkid model correctly initializes and runs speechbrain models.
We test with the test samples from common voice on CPU.
"""


import unittest
import os

from omegaconf import OmegaConf
import torch

from spkanon_eval.featex.spkid.spkid import SpkId, SAMPLE_RATE
from spkanon_eval.featex.spkid.spkid_concat import SpkIdConcat
from spkanon_eval.datamodules.collator import collate_fn
from spkanon_eval.datamodules.dataset import load_audio
from spkanon_eval.utils import seed_everything


class TestSpkid(unittest.TestCase):
    def setUp(self):
        """
        Declare the configs for the spkid models and the data directory and seed
        everything.
        """

        seed_everything(42)
        self.cfg = OmegaConf.create(
            {
                "cls": "spkanon_eval.featex.spkid.SpkId",
                "path": "speechbrain/spkrec-xvect-voxceleb",
                "train": False,
            }
        )
        self.data_dir = "tests/data/cv-corpus-10.0-2022-07-04/en/clips"

    def test_batches(self):
        """
        - Test that the ECAPA models from Speechbrain produce embeddings of
            the correct size, and that they differ from one another.
        - Test that the xvector model from Speechbrain produces embeddings of the
            correct size.
        - Test that all three models output tensors of the right size when fed batches
            with multiple samples.
        """

        model = SpkId(self.cfg, "cpu")

        # test with batches of 1 sample
        with self.subTest("1 sample"):
            for audiofile in os.listdir(self.data_dir):
                audio = load_audio(os.path.join(self.data_dir, audiofile), SAMPLE_RATE)
                audio = torch.tensor(audio).unsqueeze(0)
                audio_len = torch.tensor([audio.shape[1]])

                emb = model.run((audio, audio_len))
                assert emb.shape == (1, 512)

        # test with one batch of 2 samples
        samples = os.listdir(self.data_dir)[:2]
        batch = list()
        spk = torch.tensor(1)
        for sample in samples:
            audio = load_audio(os.path.join(self.data_dir, sample), SAMPLE_RATE)
            batch.append([audio, spk])
        batch = collate_fn(batch)

        emb = model.run(batch)
        with self.subTest("2 samples"):
            assert emb.shape == (2, 512)

    def test_concat(self):
        """
        Ensure that the SpkIdConcat class correctly concatenates the outputs of the
        spkid models.
        """

        ecapa_cfg = self.cfg.copy()
        ecapa_cfg.path = "speechbrain/spkrec-ecapa-voxceleb"
        model_cfg = [ecapa_cfg, self.cfg]
        expected_shape = [2, 704]

        concat_cfg = OmegaConf.create(
            {"cls": "spkanon_eval.featex.spkid.SpkIdConcat", "models": model_cfg}
        )
        concat_model = SpkIdConcat(concat_cfg, "cpu")
        samples = os.listdir(self.data_dir)[:2]
        batch = list()
        spk = torch.tensor(1)
        for sample in samples:
            audio = load_audio(os.path.join(self.data_dir, sample), SAMPLE_RATE)
            batch.append([audio, spk])
        batch = collate_fn(batch)

        out = concat_model.run(batch)
        self.assertEqual(list(out.shape), expected_shape)
