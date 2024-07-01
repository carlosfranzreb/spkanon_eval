import json
import logging
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from spkanon_eval.datamodules.dataset import SpeakerIdDataset
from spkanon_eval.datamodules.collator import collate_fn


LOGGER = logging.getLogger("progress")


def setup_dataloader(config: OmegaConf, datafile: str) -> DataLoader:
    """
    Create a dataloader with the SpeakerIdDataset.
    """

    LOGGER.info(f"Creating dataloader for {datafile}")
    LOGGER.info(f"\tSample rate: {config.sample_rate}")
    LOGGER.info(f"\tBatch size: {config.batch_size}")
    LOGGER.info(f"\tNum. workers: {config.num_workers}")

    return DataLoader(
        dataset=SpeakerIdDataset(datafile, config.sample_rate),
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )


def eval_dataloader(
    config: OmegaConf, datafile: str, device: str
) -> Iterable[str, list[torch.Tensor], dict[str, str]]:
    """
    This function is called by evaluation and inference scripts. It is an
    iterator over the batches and other sample info in the given manifest.

    - The data is not shuffled, so it can be mapped to the audio file paths, which
        they require to generate their results/reports.
    - Return all additional data found in the manifest file, if any. This can be the
        gender of the speaker, for example.
    """
    LOGGER.info(f"Creating eval. DL for `{datafile}`")

    # initialize the dataloader and the iterator object for the sample data
    dl = setup_dataloader(config, datafile)
    data_iter = data_iterator(datafile)

    # iterate over the batches in the dataloader
    for batch in dl:
        batch = [b.to(device) for b in batch]
        data = list()  # additional data to be returned
        # read as much `data` as there are samples in the batch
        while len(data) < batch[0].shape[0]:
            data.append(next(data_iter))
        # yield the batch, the datafile and the additional data
        yield datafile, batch, data


def data_iterator(datafile: str) -> Iterable[dict]:
    """Iterate over the JSON objects in the given datafile."""
    with open(datafile) as f:
        for line in f:
            yield json.loads(line)
