import os
import logging
import copy

import torch

from spkanon_eval.setup_module import setup
from spkanon_eval.utils import seed_everything


SAMPLE_RATE = 16000  # sample rate for evaluation
LOGGER = logging.getLogger("progress")


def evaluate(exp_folder, model, config):
    """
    Evaluate the given experiment with the components defined in the config.

    - If a new seed is defined on the eval config, set it.
    - If a component requires training, do so before starting the evaluation.
    - The evaluation is performed by the eval_dir method of each component. It receives
        as input the path to the experiment folder, where it will store its results. It
        may also receive an evaluation dataloader, if the batch size is defined in its
        config.
    """

    # change the RNG seed if required
    if "seed" in config.eval.config:
        seed_everything(config.seed)

    # ensure that the directory where the evaluation results will be stored exists
    os.makedirs(os.path.join(exp_folder, "eval"), exist_ok=True)

    # change the config params to match the anonymized data
    data_cfg = copy.deepcopy(config.data.config)
    data_cfg.sample_rate = SAMPLE_RATE

    # use the anonymized datafiles if we are not evaluating the baseline
    if config.eval.config.baseline is False:
        datafiles = [
            os.path.join(exp_folder, "results", f) for f in config.data.datasets.eval
        ]
    # if we are evaluating the baseline, use the original datafiles
    else:
        datafiles = [os.path.join(exp_folder, f) for f in config.data.datasets.eval]

    # iterate over the components and train & evaluate them
    for name, cfg in config.eval.components.items():
        cfg.data = {"config": copy.deepcopy(data_cfg)}
        component = setup(cfg, config.device, model=model)

        if cfg.train is True:
            LOGGER.info(f"Training component `{name}`")
            train_datafiles = [
                os.path.join(exp_folder, f) for f in config.data.datasets.train_eval
            ]
            component.train(exp_folder, train_datafiles)

        LOGGER.info(f"Running evaluation with component `{name}`")
        component.eval_dir(exp_folder, datafiles, config.eval.config.baseline)
        torch.cuda.empty_cache()  # delete PyTorch cache
