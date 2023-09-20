import json
import os
import logging
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from spkanon_eval.datamodules.dataset import SpeakerIdDataset
from spkanon_eval.datamodules.collator import collate_fn


LOGGER = logging.getLogger("progress")


def setup_dataloader(config: OmegaConf, datafiles: list[str]) -> DataLoader:
    """
    Create a dataloader with the SpeakerIdDataset.
    """

    LOGGER.info(f"Creating dataloader for {datafiles}")
    LOGGER.info(f"Dataloader config: {config}")
    LOGGER.info(f"Batch size: {config.batch_size}")

    return DataLoader(
        dataset=SpeakerIdDataset(datafiles, config.sample_rate),
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        drop_last=config.get("drop_last", False),
        num_workers=config.num_workers,
    )


def eval_dataloader(
    config: OmegaConf, datafiles: list[str], device: str
) -> Iterable[str, list[torch.Tensor], dict[str, str]]:
    """
    This function is called by evaluation and inference scripts. It is an
    iterator over the batches and other sample info in the given manifest.

    - The data is not shuffled, so it can be mapped to the audio file paths, which
        they require to generate their results/reports.
    - Data augmentation, if present, is also discarded.
    - If the accelerator is "gpu", the batch is moved to "cuda".
    - A different collate_fn is used, where smaller sequences are padded with zeros
        instead of repeated. Single samples can be retrieved from the batch by
        trimming the audio. The collate_fn discards the text tokens: batches consist
        only of audio tensors and their lengths.
    - Return all additional data found in the manifest file, if any. This can be the
        gender of the speaker, for example.
    """
    LOGGER.info(f"Creating eval. DL for `{datafiles}`")

    # iterate over the files in the dataloader
    for datafile in datafiles:
        # initialize the dataloader and the iterator object for the sample data
        dl = setup_dataloader(config, [datafile])
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


def data_iterator(datafile) -> Iterable[dict]:
    """
    Iterate over the JSON objects in the given manifest, and return for each
    of them the given keys.
    """
    with open(datafile) as f:
        for line in f:
            yield json.loads(line)
