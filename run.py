from argparse import ArgumentParser
import os
import logging
import yaml
from time import time
import subprocess

import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.main import main


def setup(args):
    config = load_subconfigs(yaml.load(open(args.config)))
    config = OmegaConf.create(config)
    config.resources.update(
        {
            "device": args.device,
            "n_devices": int(args.n_devices),
            "n_nodes": int(os.environ["SLURM_NNODES"])
            if "SLURM_NNODES" in os.environ
            else 1,
            "debug": args.debug,
        }
    )
    config.commit_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )

    # create the logging directory
    log_dir = os.path.join(config.log_dir, str(int(time())))
    os.makedirs(log_dir)

    # if a seed is specified, set it
    if config.seed is not None:
        pl.seed_everything(config.seed)

    # dump config file to experiment folder
    OmegaConf.save(config, os.path.join(log_dir, "exp_config.yaml"))

    # create logger in experiment folder to log progress: dump to file and stdout
    for log_file in ["progress", "targets"]:
        logger = logging.getLogger(log_file)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{log_file}.log"))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    logging.getLogger("progress").addHandler(logging.StreamHandler())

    return config, log_dir


def load_subconfigs(config):
    """
    Given a config, load all the subconfigs that are specified in the config into
    the same level as the parameter. Configs are specified by parameters ending with
    '_cfg'. If a value of the config is a dict, call this function again recursively.
    Delete the '_cfg' parameter after loading the config it points to, except those
    of the data and trainer configs.
    """

    full_config = dict()
    for key, value in config.items():
        if isinstance(value, dict):
            full_config[key] = load_subconfigs(value)
        elif key.endswith("_cfg"):
            full_config.update(load_subconfigs(yaml.load(open(value))))
        else:
            full_config[key] = value

    return full_config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_devices", default="1")
    parser.add_argument("--debug", action="store_true")
    main(*setup(parser.parse_args()))
