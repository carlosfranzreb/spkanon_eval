from argparse import ArgumentParser
import os
import logging
import yaml
from time import time, sleep
import subprocess

from omegaconf import OmegaConf

from spkanon_eval.main import main
from spkanon_eval.utils import seed_everything


def setup(args):
    config = load_subconfigs(yaml.full_load(open(args.config)))
    config = OmegaConf.create(config)
    config.device = args.device
    config.data.config.num_workers = args.num_workers
    config.commit_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )

    # create the logging directory
    log_dir = os.path.join(config.log_dir, str(int(time())))
    while os.path.exists(log_dir):
        sleep(1)
        log_dir = os.path.join(config.log_dir, str(int(time())))
    os.makedirs(log_dir)

    # if a seed is specified, set it
    if config.seed is not None:
        seed_everything(config.seed)

    # dump config file to experiment folder
    OmegaConf.save(config, os.path.join(log_dir, "exp_config.yaml"))

    # create logger in experiment folder to log progress: dump to file and stdout
    logger_name = "progress"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_name}.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    return config, log_dir


def load_subconfigs(config):
    """
    Given a config, load all the subconfigs that are specified in the config into
    the same level as the parameter. Configs are specified by parameters ending with
    '_cfg'. If a value of the config is a dict, call this function again recursively.
    """

    full_config = dict()
    for key, value in config.items():
        if isinstance(value, dict):
            full_config[key] = load_subconfigs(value)
        elif key.endswith("_cfg"):
            full_config.update(load_subconfigs(yaml.full_load(open(value))))
        else:
            full_config[key] = value

    return full_config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=10, type=int)
    main(*setup(parser.parse_args()))
