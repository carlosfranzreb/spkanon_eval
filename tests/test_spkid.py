"""
Test that the spkid model correctly initializes and runs both speechbrain and nemo
models. We test with the test samples from common voice on CPU.
"""


import unittest
import os

from omegaconf import OmegaConf
import torch
import librosa
from pytorch_lightning import seed_everything

from src.featex.spkid.spkid import SpkId, SAMPLE_RATE
from src.featex.spkid.spkid_concat import SpkIdConcat
from src.dataloader import collate_fn_inference


class TestSpkid(unittest.TestCase):
    def setUp(self):
        """
        Declare the configs for the spkid models and the data directory and seed
        everything.
        """

        seed_everything(42)
        self.nemo_cfg = OmegaConf.create(
            {
                "cls": "src.featex.spkid.SpkId",
                "toolkit": "nemo",
                "path": "ecapa_tdnn",
                "train": False,
            }
        )
        self.sb_ecapa_cfg = self.nemo_cfg.copy()
        self.sb_ecapa_cfg.toolkit = "speechbrain"
        self.sb_ecapa_cfg.path = "speechbrain/spkrec-ecapa-voxceleb"
        self.sb_xvector_cfg = self.sb_ecapa_cfg.copy()
        self.sb_xvector_cfg.path = "speechbrain/spkrec-xvect-voxceleb"

        self.data_dir = "tests/data/cv-corpus-10.0-2022-07-04/en/clips"

    def test_configs(self):
        """
        - Test that the ECAPA models from NeMo and Speechbrain produce embeddings of
            the correct size, and that they differ from one another.
        - Test that the xvector model from Speechbrain produces embeddings of the
            correct size.
        - Test that all three models output tensors of the right size when fed batches
            with multiple samples.
        """

        nemo_model = SpkId(self.nemo_cfg, "cpu")
        sb_ecapa_model = SpkId(self.sb_ecapa_cfg, "cpu")
        sb_xvector_model = SpkId(self.sb_xvector_cfg, "cpu")

        # test with batches of 1 sample
        with self.subTest("1 sample"):
            for audiofile in os.listdir(self.data_dir):

                audio, _ = librosa.load(
                    os.path.join(self.data_dir, audiofile), sr=SAMPLE_RATE
                )
                audio = torch.tensor(audio).unsqueeze(0)
                audio_len = torch.tensor([audio.shape[1]])

                nemo_emb = nemo_model.run((audio, audio_len))
                sb_ecapa_emb = sb_ecapa_model.run((audio, audio_len))
                sb_xvector_emb = sb_xvector_model.run((audio, audio_len))

                assert nemo_emb.shape == (1, 192)
                assert sb_ecapa_emb.shape == (1, 192)
                assert sb_xvector_emb.shape == (1, 512)
                assert torch.all(nemo_emb != sb_ecapa_emb)

        # test with one batch of 2 samples
        samples = os.listdir(self.data_dir)[:2]
        batch = list()
        t = torch.tensor(1)
        for sample in samples:
            audio, _ = librosa.load(os.path.join(self.data_dir, sample), sr=SAMPLE_RATE)
            batch.append([torch.tensor(audio), torch.tensor(audio.shape[0]), t, t])
        batch = collate_fn_inference(batch)

        nemo_emb = nemo_model.run(batch)
        sb_ecapa_emb = sb_ecapa_model.run(batch)
        sb_xvector_emb = sb_xvector_model.run(batch)

        with self.subTest("2 samples"):
            assert nemo_emb.shape == (2, 192)
            assert sb_ecapa_emb.shape == (2, 192)
            assert sb_xvector_emb.shape == (2, 512)
            assert torch.all(nemo_emb != sb_ecapa_emb)

    def test_spkid_concat(self):
        """
        Ensure that the SpkIdConcat class correctly concatenates the outputs of the
        spkid models.
        """

        scenarios = [
            [[self.sb_ecapa_cfg, self.sb_xvector_cfg], [2, 704]],
            [[self.sb_ecapa_cfg, self.nemo_cfg], [2, 384]],
        ]

        for models_cfg, expected_shape in scenarios:
            concat_cfg = OmegaConf.create(
                {"cls": "src.featex.spkid.SpkIdConcat", "models": models_cfg}
            )
            concat_model = SpkIdConcat(concat_cfg, "cpu")
            samples = os.listdir(self.data_dir)[:2]
            batch = list()
            t = torch.tensor(1)
            for sample in samples:
                audio, _ = librosa.load(
                    os.path.join(self.data_dir, sample), sr=SAMPLE_RATE
                )
                batch.append([torch.tensor(audio), torch.tensor(audio.shape[0]), t, t])
            batch = collate_fn_inference(batch)

            out = concat_model.run(batch)
            self.assertEqual(list(out.shape), expected_shape, msg=models_cfg)
