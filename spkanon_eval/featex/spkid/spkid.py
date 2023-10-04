"""
Wrapper for Speechbrain speaker recognition models. The `run` method returns speaker
embeddings.
"""


import os
import logging
import json
import csv
import random

from speechbrain.pretrained import EncoderClassifier
from hyperpyyaml import load_hyperpyyaml
from omegaconf import OmegaConf
import torch

from spkanon_eval.featex.spkid.finetune import SpeakerBrain, prepare_dataset


LOGGER = logging.getLogger("progress")
SAMPLE_RATE = 16000


class SpkId:
    def __init__(self, config: OmegaConf, device: str) -> None:
        """Initialize the model with the given config and freeze its parameters."""
        self.config = config
        self.device = device
        self.save_dir = os.path.join("checkpoints", config.path)
        self.model = EncoderClassifier.from_hparams(
            source=config.path, savedir=self.save_dir, run_opts={"device": device}
        )
        if config.emb_model_ckpt is not None:
            LOGGER.info(f"Loading emb. model from {config.emb_model_ckpt}")
            state_dict = torch.load(config.emb_model_ckpt, map_location=device)
            self.model.mods.embedding_model.load_state_dict(state_dict)
        self.model.eval()

    def run(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """
        Return speaker embeddings for the given batch of utterances.

        Args:
            batch: A list of two tensors, the first containing the waveforms
            with shape (batch_size, n_samples), and the second containing
            the speaker labels as integers with shape (batch_size).

        Returns:
            A tensor containing the speaker embeddings with shape
            (batch_size, embedding_dim).
        """
        return self.model.encode_batch(batch[0].to(self.device)).squeeze(1)

    def finetune(self, dump_dir: str, datafiles: list[str], n_speakers: int) -> None:
        """
        Fine-tune this model with the given datafiles.

        Args:
            dump_dir: Path to the folder where the model and datafiles will be saved.
            datafiles: List of paths to the datafiles used for fine-tuning.
            n_speakers: Number of speakers across all datafiles, used to initialize
                the classifier.
        """

        LOGGER.info(f"Fine-tuning the spkid model with datafiles {datafiles}")

        with open(self.config.finetune_config) as f:
            hparams = load_hyperpyyaml(
                f,
                overrides={
                    "output_folder": dump_dir,
                    "out_n_neurons": n_speakers,
                    "num_workers": self.config.num_workers,
                },
            )

        # create the datafiles and writers
        os.makedirs(dump_dir, exist_ok=True)
        splits = dict()
        for split in ["train", "val"]:
            splits[split] = dict()
            splits[split]["file"] = os.path.join(dump_dir, f"{split}.csv")
            splits[split]["writer"] = open(splits[split]["file"], "w")
            splits[split]["csv_writer"] = csv.writer(
                splits[split]["writer"],
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            splits[split]["csv_writer"].writerow(
                ["ID", "duration", "wav", "spk_id", "spk_id_encoded"]
            )

        # split the data and store the speaker IDs
        speaker_ids = list()
        for datafile in datafiles:
            speaker_objs = list()
            for line in open(datafile):
                obj = json.loads(line)
                if obj["label"] not in speaker_ids:
                    speaker_ids.append(obj["label"])
                    if len(speaker_objs) > 0:
                        split_spk_utts(
                            speaker_objs,
                            splits["train"]["csv_writer"],
                            splits["val"]["csv_writer"],
                            hparams["val_ratio"],
                            len(speaker_ids) - 2,
                        )
                        speaker_objs = list()
                speaker_objs.append(obj)
            speaker_ids.append(speaker_objs[0]["label"])
            split_spk_utts(
                speaker_objs,
                splits["train"]["csv_writer"],
                splits["val"]["csv_writer"],
                hparams["val_ratio"],
                len(speaker_ids) - 2,
            )

        for split in splits:
            splits[split]["writer"].close()
        json.dump(speaker_ids, open(os.path.join(dump_dir, "spk_ids.json"), "w"))

        # train the model
        train_data = prepare_dataset(hparams, splits["train"]["file"])
        val_data = prepare_dataset(hparams, splits["val"]["file"])
        speaker_brain = SpeakerBrain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts={"device": self.device},
        )
        speaker_brain.epoch_losses = {"TRAIN": [], "VALID": []}
        val_kwargs = hparams["dataloader_options"]
        val_kwargs["shuffle"] = False
        speaker_brain.fit(
            speaker_brain.hparams.epoch_counter,
            train_data,
            val_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=val_kwargs,
        )

        # save the embedding model and load it
        emb_model_state_dict = speaker_brain.modules.embedding_model.state_dict()
        torch.save(emb_model_state_dict, os.path.join(dump_dir, "embedding_model.pt"))
        self.model.mods.embedding_model.load_state_dict(emb_model_state_dict)


def split_spk_utts(
    speaker_objs: list[dict],
    train_writer: csv.writer,
    val_writer: csv.writer,
    ratio: float,
    spk_id: int,
) -> None:
    """
    Split the given list of speaker utterances into training and validation sets.

    Args:
        speaker_objs: List of speaker utterances from the datafile.
        train_writer: CSV writer for the training datafile.
        val_writer: CSV writer for the validation datafile.
        ratio: Ratio of validation utterances.
        spk_id: Speaker ID, as stored in self.speakers.
    """

    indices = list(range(len(speaker_objs)))
    random_indices = random.sample(indices, len(indices))
    n_val = int(len(speaker_objs) * ratio)

    for idx, random_idx in enumerate(random_indices):
        obj = speaker_objs[random_idx]
        writer = val_writer if idx < n_val else train_writer
        writer.writerow(
            [
                obj["audio_filepath"],
                obj["duration"],
                obj["audio_filepath"],
                obj["label"],
                spk_id,
            ]
        )
