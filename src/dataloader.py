import json
import os
import logging

import torch
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_label import AudioToSpeechLabelDataset
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.neural_types import *


LOGGER = logging.getLogger("progress")


def setup_dataloader(config, manifest_filepath, collate_fn=None):
    """
    Create a NeMo dataloader with speaker labels.
    """

    LOGGER.info(f"Creating dataloader for {manifest_filepath}")
    LOGGER.info(f"Dataloader config: {config}")
    LOGGER.info(f"Batch size: {config.trainer.batch_size}")
    LOGGER.info(f"Collate function: {collate_fn}")

    if "augmentor" in config:
        augmentor = process_augmentations(config.augmentor)
    else:
        augmentor = None
    featurizer = WaveformFeaturizer(
        sample_rate=config.sample_rate,
        int_values=config.get("int_values", False),
        augmentor=augmentor,
    )

    dataset = AudioToSpeechLabelDataset(
        manifest_filepath=manifest_filepath,
        labels=get_labels(manifest_filepath),
        featurizer=featurizer,
        trim=config.get("trim_silence", False),
        normalize_audio=config.get("normalize_audio", False),
        cal_labels_occurrence=config.get("cal_labels_occurrence", False),
    )

    # pick collate_fn if needed; this one repeats smaller sequences
    if collate_fn is None:
        if hasattr(dataset, "fixed_seq_collate_fn"):
            collate_fn = dataset.fixed_seq_collate_fn
        else:
            collate_fn = dataset.datasets[0].fixed_seq_collate_fn

    return DataLoader(
        dataset=dataset,
        batch_size=config.trainer.batch_size,
        collate_fn=collate_fn,
        drop_last=config.get("drop_last", False),
        shuffle=config.get("shuffle", False),
        num_workers=config.trainer.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )


def get_labels(manifest_filepath):
    """Return a list containing all speaker labels present in the file(s)."""
    labels = list()
    if isinstance(manifest_filepath, str):
        manifest_filepath = [manifest_filepath]
    for fpath in manifest_filepath:
        with open(fpath) as f:
            for line in f:
                obj = json.loads(line)
                if obj["label"] not in labels:
                    labels.append(obj["label"])
    return labels


def add_root(manifest_filepath, root_folder, log_dir, min_dur=None, max_dur=None):
    """
    Add the root folder to the paths of the audiofiles and store the resulting
    manifest files within the experiment folder.
    """
    LOGGER.info(f"Adding root `{root_folder}` to datafiles `{manifest_filepath}`")
    new_paths = list()
    if isinstance(manifest_filepath, str):
        manifest_filepath = [manifest_filepath]
    for fpath in manifest_filepath:
        LOGGER.info(f"Adding root `{root_folder}` to `{fpath}`")
        new_paths.append(os.path.join(log_dir, fpath))
        if os.path.exists(new_paths[-1]):  # manifest already modified
            LOGGER.info(f"New filepath `{new_paths[-1]}` already exists; skipping")
            continue
        os.makedirs(os.path.dirname(new_paths[-1]), exist_ok=True)
        too_short, too_long, included = list(), list(), list()
        with open(new_paths[-1], "w") as writer:
            with open(fpath) as reader:
                for line in reader:
                    obj = json.loads(line)
                    dur = obj["duration"]
                    if min_dur is not None and dur < min_dur:
                        too_short.append(dur)
                        continue
                    if max_dur is not None and dur > max_dur:
                        too_long.append(dur)
                        continue
                    included.append(dur)
                    obj["audio_filepath"] = obj["audio_filepath"].replace(
                        "{root}", root_folder
                    )
                    writer.write(json.dumps(obj) + "\n")

        LOGGER.info(
            f"{len(included)} samples included ({round(sum(included) / 3600, 3 )} h)"
        )
        LOGGER.info(
            f"{len(too_short)} samples too short ({round(sum(too_short) / 3600, 3 )} h)"
        )
        LOGGER.info(
            f"{len(too_long)} samples too long ({round(sum(too_long) / 3600, 3 )} h)"
        )

    LOGGER.info(f"New datafiles: `{new_paths}`")
    return new_paths


def eval_dataloader(config, manifest_filepath, device):
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
    LOGGER.info(f"Creating eval. DL for `{manifest_filepath}`")
    config.shuffle = False  # important to map filenames to samples
    if "augmentor" in config:
        config.augmentor = None  # no augmentation during inference

    if isinstance(manifest_filepath, str):  # datafiles must be in a list
        manifest_filepath = [manifest_filepath]

    # iterate over the files in the dataloader
    for fpath in manifest_filepath:
        # initialize the dataloader and the iterator object for the sample data
        dl = setup_dataloader(config, fpath, collate_fn_inference)
        data_iter = data_iterator(fpath)

        # iterate over the batches in the dataloader
        for batch in dl:
            batch = [b.to(device) for b in batch]
            data = list()  # additional data to be returned
            # read as much `data` as there are samples in the batch
            while len(data) < batch[0].shape[0]:
                data.append(next(data_iter))
            # yield the batch, the datafile and the additional data
            yield fpath, batch, data


def collate_fn_inference(batch):
    """
    - Pads the shorter audio tensors with zeros.
    - Lengths, labels and labels_lengths are not modified.
    - If one of the audios has multiple channels, only the first one is kept.
    """
    _, audio_lengths, labels, labels_lengths = zip(*batch)
    max_audio_len = max(audio_lengths).item()

    audio_signal = []
    for sig, sig_len, _, _ in batch:
        if sig.ndim > 1:
            sig = sig[:, 0]
        sig_len = sig_len.item()
        if sig_len < max_audio_len:
            pad = (0, max_audio_len - sig_len)
            sig = torch.nn.functional.pad(sig, pad)
        audio_signal.append(sig)

    audio_signal = torch.stack(audio_signal)
    audio_lengths = torch.stack(audio_lengths)
    labels = torch.stack(labels)
    labels_lengths = torch.stack(labels_lengths)
    return audio_signal, audio_lengths, labels, labels_lengths


def data_iterator(datafile):
    """
    Iterate over the JSON objects in the given manifest, and return for each
    of them the given keys.
    """
    with open(datafile) as f:
        for line in f:
            yield json.loads(line)
