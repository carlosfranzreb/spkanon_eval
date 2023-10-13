import os
import logging

from spkanon_eval.anonymizer import Anonymizer
from spkanon_eval.inference import infer
from spkanon_eval.evaluate import evaluate
from spkanon_eval.datamodules.utils import prepare_datafile


LOGGER = logging.getLogger("progress")
TARGET_LOGGER = logging.getLogger("targets")


def main(config, log_dir):
    """Run the jobs (training, inference, evaluation) as specified in the config."""

    model = Anonymizer(config, log_dir)  # create the model

    if "inference" in config and config.inference.run is not False:
        # if an experiment folder is given, use it, otherwise use the current one
        if isinstance(config.inference.run, str):  # path is given
            exp_folder = config.inference.run
        else:  # infer an existing exp folder
            exp_folder = log_dir
        log_msg = f"### Start of inference with experiment folder `{exp_folder}`"
        LOGGER.info(log_msg)
        TARGET_LOGGER.info(log_msg)
        # create the eval datafiles if they don't exist
        if not os.path.exists(os.path.join(exp_folder, "data", "eval.txt")):
            prepare_datafile("eval", config, exp_folder)
        infer(exp_folder, "eval", model, config)
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
            prepare_datafile("train_eval", config, exp_folder)

        # create the eval datafiles if they don't exist (for the baseline)
        if not os.path.exists(os.path.join(exp_folder, "data", "eval.txt")):
            prepare_datafile("eval", config, exp_folder)

        evaluate(exp_folder, model, config)
        LOGGER.info("End of evaluation")
