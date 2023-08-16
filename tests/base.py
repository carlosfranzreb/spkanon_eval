"""
This base class should be inherited by all test classes.
- It provides a few helper methods and a setUp and tearDown that start
    and experiment and delete it at the end.
"""


import unittest
import os
from argparse import ArgumentParser
from shutil import rmtree

from omegaconf import OmegaConf

from run import setup
from src.main import main


MARKER_FILE = ".TEST-FOLDER"
LOG_DIR = "tests/logs"
DUMP_CONFIG = "config.yaml"


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
                "train": False,
                "inference": {
                    "run": True,
                    "input": {"spectrogram": "spectrogram", "target": "target"},
                },
                "seed": 0,
                "log_dir": LOG_DIR,
                "trainer": {
                    "batch_size": 2,
                    "max_epochs": 20,
                    "num_workers": 0,
                    "accumulate_grad_batches": 1,
                    "limit_train_batches": 1.0,
                    "limit_val_batches": 1.0,
                },
                "sample_rate": 24000,
                "featex": {
                    "spectrogram": {
                        "cls": "src.featex.spectrogram.SpecExtractor",
                        "n_mels": 80,
                        "n_fft": 2048,
                        "win_length": 1200,
                        "hop_length": 300,
                    }
                },
                "synthesis": {
                    "cls": "src.synthesis.dummy.DummySynthesizer",
                    "sample_rate": "${sample_rate}",
                    "input": {"spectrogram": "spectrogram"},
                },
                "data": {
                    "config": {
                        "trainer": "${trainer}",
                        "root_folder": "tests/data",
                        "sample_rate": "${sample_rate}",
                        "trim_silence": False,
                        "shuffle": False,
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
                "resources": {
                    "device": "cpu",
                    "n_devices": 1,
                    "n_nodes": 1,
                    "debug": False,
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
        """Delete the logging directory."""
        rmtree(self.log_dir)


def run_pipeline(config):
    """
    Run the pipeline with the current config.
    """
    args = ArgumentParser()

    # save the config to a file and pass the file to the setup function
    OmegaConf.save(config, DUMP_CONFIG)
    args.config = DUMP_CONFIG

    args.device = config.resources.device
    args.n_devices = config.resources.n_devices
    args.debug = config.resources.debug
    config, log_dir = setup(args)
    main(config, log_dir)
    return config, log_dir
