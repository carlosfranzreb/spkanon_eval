import os
import logging
import shutil

from omegaconf import OmegaConf

from spkanon_eval.anonymizer import Anonymizer
from spkanon_eval.inference import infer
from spkanon_eval.evaluate import evaluate
from spkanon_eval.datamodules.utils import prepare_datafile


LOGGER = logging.getLogger("progress")


def main(config: OmegaConf, exp_folder: str):
    """Run the jobs (training, inference, evaluation) as specified in the config."""

    model = Anonymizer(config, exp_folder)  # create the model

    if "inference" in config and config.inference.run is not False:
        log_msg = f"### Start of inference with experiment folder `{exp_folder}`"
        LOGGER.info(log_msg)
        # create the eval datafiles if they don't exist
        if not os.path.exists(os.path.join(exp_folder, "data", "eval.txt")):
            prepare_datafile("eval", config, exp_folder)
        infer(exp_folder, "eval", model, config)
        LOGGER.info("End of inference")

    if "eval" in config:
        log_msg = "### Start of evaluation"
        LOGGER.info(log_msg)
        # if an experiment folder is given, copy its eval datafiles
        if config.eval.config.exp_folder is not None:
            os.makedirs(os.path.join(exp_folder, "data"), exist_ok=True)
            files = ["eval"]
            if config.eval.config.baseline is False:
                files.append("anon_eval")
            if any([c.train for c in config.eval.components.values()]):
                files.append("train_eval")
                if os.path.exists(
                    os.path.join(config.eval.config.exp_folder, "anon_train_eval.txt")
                ):
                    files.append("anon_train_eval")
            LOGGER.info(
                f"Copying datafiles {files} from {config.eval.config.exp_folder}"
            )
            for f in files:
                shutil.copy(
                    os.path.join(config.eval.config.exp_folder, "data", f + ".txt"),
                    os.path.join(exp_folder, "data", f + ".txt"),
                )
            config.data.config.anon_folder = config.eval.config.exp_folder

        elif any([c.train for c in config.eval.components.values()]):
            prepare_datafile("train_eval", config, exp_folder)

        # create the eval datafiles if they don't exist (for the baseline)
        if not os.path.exists(os.path.join(exp_folder, "data", "eval.txt")):
            prepare_datafile("eval", config, exp_folder)

        evaluate(exp_folder, model, config)
        LOGGER.info("End of evaluation")
