"""
This base class should be inherited by all test classes.
- It provides a few helper methods and a setUp and tearDown that start
    and experiment and delete it at the end.
"""


import os
import unittest
from argparse import ArgumentParser
from shutil import rmtree
from tempfile import NamedTemporaryFile

from omegaconf import OmegaConf

from run import setup
from spkanon_eval.main import main


MARKER_FILE = ".TEST-FOLDER"
LOG_DIR = "tests/logs"


class BaseTestClass(unittest.TestCase):
    def setUp(self):
        """
        Create the config object with the StarGANv2-VC pipeline. Training is disabled,
        inference and evaluation are enabled.
        It will be modified by the test class before running the experiment with
        `run_pipeline()`. The logging directory will be stored in `self.log_dir`, and
        deleted after the test with `tearDown()`.
        """

        self.init_config = OmegaConf.create(
            {
                "name": "StarGANv2-VC",
                "inference": {
                    "run": True,
                    "input": {"spectrogram": "spectrogram", "target": "target"},
                },
                "seed": 0,
                "log_dir": LOG_DIR,
                "device": "cpu",
                "sample_rate": 24000,
                "batch_size": 2,
                "featex": {
                    "spectrogram": {
                        "cls": "spkanon_eval.featex.spectrogram.SpecExtractor",
                        "n_mels": 80,
                        "n_fft": 2048,
                        "win_length": 1200,
                        "hop_length": 300,
                    }
                },
                "synthesis": {
                    "cls": "spkanon_eval.synthesis.dummy.DummySynthesizer",
                    "sample_rate": "${sample_rate}",
                    "input": {"spectrogram": "spectrogram"},
                },
                "data": {
                    "config": {
                        "root_folder": "tests/data",
                        "sample_rate": "${sample_rate}",
                        "batch_size": "${batch_size}",
                    },
                    "datasets": {
                        "eval": [
                            "data/debug/edacc-test.txt",
                            "data/debug/ls-dev-clean-2.txt",
                            "data/debug/cv-test.txt",
                            "data/debug/ravdess.txt",
                        ],
                        "train_eval": ["data/debug/ls-dev-clean-2.txt"],
                    },
                },
                "eval": {
                    "config": {
                        "baseline": False,
                        "exp_folder": None,
                        "sample_rate": "${synthesis.sample_rate}",
                    },
                    "components": dict(),  # will be filled by the test class
                },
            }
        )

    def tearDown(self):
        """Delete the logging directory if it exists as an attribute."""
        if hasattr(self, "log_dir"):
            rmtree(self.log_dir)


def run_pipeline(config):
    """
    Run the pipeline with the current config.
    """
    args = ArgumentParser()
    args.device = config.device
    args.num_workers = 0

    # save the config to a file and pass the file to the setup function
    with NamedTemporaryFile(mode="w+", encoding="utf-8") as tmp_file:
        OmegaConf.save(config, tmp_file.name)
        args.config = tmp_file.name
        config, log_dir = setup(args)

    main(config, log_dir)
    return config, log_dir
