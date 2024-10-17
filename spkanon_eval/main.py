import os
import logging
import shutil
import json
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from spkanon_eval.anonymizer import Anonymizer
from spkanon_eval.inference import infer
from spkanon_eval.evaluate import evaluate
from spkanon_eval.datamodules import prepare_datafile


LOGGER = logging.getLogger("progress")


def main(config: OmegaConf, exp_folder: str):
    """Run the jobs (training, inference, evaluation) as specified in the config."""

    # Otherwise SB assumes we are running a distributed job
    if "LOCAL_RANK" in os.environ:
        del os.environ["LOCAL_RANK"]

    target_df = config.data.datasets.get("targets", list())
    if len(target_df) > 0:
        prepare_datafile("targets", config, exp_folder)

    model = Anonymizer(config, exp_folder)
    config.data.config.chunk_sizes = dict()

    if "inference" in config and config.inference.run is not False:
        LOGGER.info(f"### Start of inference with experiment folder `{exp_folder}`")
        # create the eval datafiles if they don't exist
        if not os.path.exists(os.path.join(exp_folder, "data", "eval.txt")):
            df = prepare_datafile("eval", config, exp_folder)
            config.data.config.chunk_sizes["eval"] = compute_chunk_sizes(
                df, model, config.sample_rate
            )

        infer(exp_folder, "eval", model, config)
        LOGGER.info("End of inference")

    if "eval" in config:
        LOGGER.info("### Start of evaluation")
        # if an experiment folder is given, copy its eval datafiles
        if config.eval.config.exp_folder is not None:
            os.makedirs(os.path.join(exp_folder, "data"), exist_ok=True)
            files = ["eval"]
            if config.eval.config.baseline is False:
                files.append("anon_eval")
            if any([c.train for c in config.eval.components.values()]):
                files.append("train_eval")
                if os.path.exists(
                    os.path.join(
                        config.eval.config.exp_folder, "data", "anon_train_eval.txt"
                    )
                ):
                    files.append("anon_train_eval")
            LOGGER.info(
                f"Copying datafiles {files} from {config.eval.config.exp_folder}"
            )
            for f in files:
                f_dst = os.path.join(exp_folder, "data", f + ".txt")
                shutil.copy(
                    os.path.join(config.eval.config.exp_folder, "data", f + ".txt"),
                    f_dst,
                )
                if "anon" not in f:
                    config.data.config.chunk_sizes[f] = compute_chunk_sizes(
                        f_dst, model, config.sample_rate
                    )
            config.data.config.anon_folder = config.eval.config.exp_folder

        elif any([c.train for c in config.eval.components.values()]):
            prepare_datafile("train_eval", config, exp_folder)

        # create the eval datafiles if they don't exist (for the baseline)
        if not os.path.exists(os.path.join(exp_folder, "data", "eval.txt")):
            prepare_datafile("eval", config, exp_folder)

        evaluate(exp_folder, model, config)
        LOGGER.info("End of evaluation")


def compute_chunk_sizes(
    datafile: str, model, sample_rate: int, n_chunks: int = 10
) -> dict:
    """
    Compute the chunk sizes for the given datafile. The chunk size determines the
    number of samples that are included in a batch depending on their maximum duration.

    Args:
        datafile: path to the datafile
        model: the model for which the chunk sizes are computed
        n_chunks: the number of chunks to compute

    Returns:
        A dictionary mapping the maximum duration of a batch to the number of samples
        in the batch.
    """
    LOGGER.info(f"Computing chunk sizes for file {datafile} and model {model}")

    sorted_durs = sorted([json.loads(line)["duration"] for line in open(datafile)])
    min_dur, max_dur = sorted_durs[0], sorted_durs[-1]

    if model.device == "cpu":
        LOGGER.warning("Model is on CPU. Skipping chunk size computation.")
        return {max_dur: 1}

    chunk_sizes = dict()
    batch_size = 1
    for chunk_max_dur in torch.linspace(max_dur, min_dur, n_chunks):
        n_samples = int(chunk_max_dur.item() * sample_rate)
        while True:
            batch = [
                torch.randn([batch_size, n_samples], device=model.device),
                torch.randint(10, [batch_size], device=model.device),
                torch.ones(batch_size, device=model.device, dtype=torch.int32)
                * n_samples,
            ]

            try:
                if hasattr(model, "forward"):
                    data = [
                        {"speaker_id": val.item(), "gender": True} for val in batch[1]
                    ]
                    model.forward(batch, data)
                else:
                    model.run(batch)
                batch_size += 1
            except torch.cuda.OutOfMemoryError:
                break

        chunk_sizes[chunk_max_dur.item()] = batch_size


def seed_everything(seed: int):
    """Set the seed for Python, Numpy and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
