"""
Test that the spkid model correctly initializes and runs speechbrain models.
We test with the test samples from common voice on CPU.
"""


import unittest
import os
import shutil
import json
import copy

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
                "path": "speechbrain/spkrec-xvect-voxceleb",
                "emb_model_ckpt": None,
                "num_workers": 0,
                "finetune_config": "spkanon_eval/config/components/spkid/train_xvec_debug.yaml",
            }
        )
        self.data_dir = "spkanon_eval/tests/data/LibriSpeech/dev-clean-2/1988/24833"

    def test_batches(self):
        """
        Ensure that batching works with the x-vector model.
        """

        model = SpkId(self.cfg, "cpu")

        # test with batches of 1 sample
        with self.subTest("1 sample"):
            for audiofile in os.listdir(self.data_dir):
                audio = load_audio(os.path.join(self.data_dir, audiofile), SAMPLE_RATE)
                audio = torch.tensor(audio).unsqueeze(0)
                audio_len = torch.tensor([audio.shape[1]])

                emb = model.run((audio, audio_len))
                self.assertTrue(emb.shape == (1, 512))

        # test with one batch of 2 samples
        samples = os.listdir(self.data_dir)[:2]
        batch = list()
        spk = torch.tensor(1)
        for sample in samples:
            audio = load_audio(os.path.join(self.data_dir, sample), SAMPLE_RATE)
            batch.append([audio, spk, audio.shape[0]])
        batch = collate_fn(batch)

        emb = model.run(batch)
        with self.subTest("2 samples"):
            self.assertTrue(emb.shape == (2, 512))

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
            batch.append([audio, spk, audio.shape[0]])
        batch = collate_fn(batch)

        out = concat_model.run(batch)
        self.assertEqual(list(out.shape), expected_shape)

    def test_finetune(self):
        """
        Test that the spkid model correctly fine-tunes on the given datafile.
        ! We assume that the val_ratio is 0.4.
        """
        exp_folder = "spkanon_eval/tests/logs/spkid_finetune"
        if os.path.isdir(exp_folder):
            shutil.rmtree(exp_folder)
        os.makedirs(os.path.join(exp_folder))
        datafile = "spkanon_eval/tests/datafiles/ls-dev-clean-2.txt"
        n_speakers = 3
        model = SpkId(self.cfg, "cpu")
        old_state_dict = copy.deepcopy(model.model.state_dict())
        model.finetune(exp_folder, datafile, n_speakers)

        # gather the utterances from the datafile
        expected_samples = dict()
        for line in open(os.path.join(datafile)):
            obj = json.loads(line)
            expected_samples[obj["path"]] = obj

        split_samples = dict()

        # check the train samples
        with open(os.path.join(exp_folder, "train.csv")) as f:
            train_lines = f.readlines()
        self.assertEqual(len(train_lines), 8)
        for line in train_lines[1:]:
            split = line.strip().split(",")
            self.assertTrue(split[0] in expected_samples)
            split_samples[split[0]] = split

        # check the val samples
        with open(os.path.join(exp_folder, "val.csv")) as f:
            val_lines = f.readlines()
        self.assertEqual(len(val_lines), 4)
        for line in val_lines[1:]:
            split = line.strip().split(",")
            self.assertTrue(split[0] in expected_samples)
            split_samples[split[0]] = split

        # check that the samples are the same
        for path, split in split_samples.items():
            obj = expected_samples[path]
            for idx, key in enumerate(
                ["path", "duration", "path", "label", "speaker_id"]
            ):
                self.assertTrue(split[idx] == str(obj[key]))

        # check that all samples are used
        self.assertEqual(len(split_samples), len(expected_samples))

        # check that the model is trained one epoch
        log_file = os.path.join(exp_folder, "train_log.txt")
        self.assertTrue(os.path.isfile(log_file))
        with open(log_file) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        new_state_dict = model.model.state_dict()
        self.assertTrue(
            any(
                [
                    not torch.equal(old, new)
                    for old, new in zip(
                        old_state_dict.values(), new_state_dict.values()
                    )
                ]
            )
        )

        # check that the model is saved
        model_file = os.path.join(exp_folder, "embedding_model.pt")
        self.assertTrue(os.path.isfile(model_file))

        shutil.rmtree(exp_folder)
