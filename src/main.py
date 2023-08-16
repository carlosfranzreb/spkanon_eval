import os
import logging

import torch

from src.anonymizer import Anonymizer
from src.inference import infer
from src.evaluate import evaluate
from src.dataloader import add_root


LOGGER = logging.getLogger("progress")
TARGET_LOGGER = logging.getLogger("targets")


def main(config, log_dir):
    """Run the jobs (training, inference, evaluation) as specified in the config."""

    # if the target selection algorithm requires training, prepare the data
    if "target_selection" in config:
        train_alg = config.target_selection.get("train", False)
        if train_alg is True:
            add_root(
                config.data.datasets.train_targetselection,
                config.data.config.root_folder,
                log_dir,
                config.data.config.get("min_duration", None),
                config.data.config.get("max_duration", None),
            )

    model = Anonymizer(config, log_dir)  # create the model

    if "inference" in config and config.inference.run is not False:
        # if an experiment folder is given, use it, otherwise use the current one
        if isinstance(config.inference.run, str):  # path is given
            exp_folder = config.inference.run
        else:  # infer an existing exp folder
            exp_folder = model.log_dir
        log_msg = f"### Start of inference with experiment folder `{exp_folder}`"
        LOGGER.info(log_msg)
        TARGET_LOGGER.info(log_msg)
        # create the eval datafiles if they don't exist
        if not all(
            [
                os.path.exists(os.path.join(exp_folder, f))
                for f in config.data.datasets.eval
            ]
        ):
            add_root(
                config.data.datasets.eval,
                config.data.config.root_folder,
                exp_folder,
                config.data.config.get("min_duration", None),
                config.data.config.get("max_duration", None),
            )
        infer(
            model,
            exp_folder,
            [os.path.join(exp_folder, f) for f in config.data.datasets.eval],
            os.path.join(exp_folder, "results"),
            config.inference.input,
            config.data.config,
            config.synthesis.sample_rate,
        )
        LOGGER.info("End of inference")

    if "eval" in config:
        log_msg = "### Start of evaluation"
        LOGGER.info(log_msg)
        TARGET_LOGGER.info(log_msg)
        # if an experiment folder is given, use it, otherwise use the current one
        if config.eval.config.exp_folder is not None:
            exp_folder = config.eval.config.exp_folder
        else:
            exp_folder = model.log_dir
        LOGGER.info(f"Evaluating experiment folder `{exp_folder}`")

        # if any component requires training, create the necessary datafiles
        if any([c.train for c in config.eval.components.values()]):
            add_root(
                config.data.datasets.train_eval,
                config.data.config.root_folder,
                exp_folder,
                config.data.config.get("min_duration", None),
                config.data.config.get("max_duration", None),
            )

        # create the eval datafiles if they don't exist (for the baseline)
        if not all(
            [
                os.path.exists(os.path.join(exp_folder, f))
                for f in config.data.datasets.eval
            ]
        ):
            add_root(
                config.data.datasets.eval,
                config.data.config.root_folder,
                exp_folder,
                config.data.config.get("min_duration", None),
                config.data.config.get("max_duration", None),
            )

        evaluate(exp_folder, model, config)
        LOGGER.info("End of evaluation")
