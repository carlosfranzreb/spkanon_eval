"""
Wrapper for Speechbrain speaker recognition models. The `run` method returns speaker
embeddings.
"""


import shutil
import os
import logging
import json
import csv

from speechbrain.pretrained import EncoderClassifier
from speechbrain.utils.checkpoints import Checkpointer
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

        # create the train datafile as expected by SpeechBrain
        os.makedirs(dump_dir, exist_ok=True)
        train_datafile = os.path.join(dump_dir, "train_datafile.json")
        csv_file = open(train_datafile, "w")
        csv_writer = csv.writer(
            csv_file,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )
        speaker_ids = list()
        csv_writer.writerow(["ID", "duration", "wav", "spk_id", "spk_id_encoded"])
        for datafile in datafiles:
            for line in open(datafile):
                obj = json.loads(line)
                if obj["label"] not in speaker_ids:
                    speaker_ids.append(obj["label"])
                csv_writer.writerow(
                    [
                        obj["audio_filepath"],
                        obj["duration"],
                        obj["audio_filepath"],
                        obj["label"],
                        speaker_ids.index(obj["label"]),
                    ]
                )
        csv_file.close()
        json.dump(speaker_ids, open(os.path.join(dump_dir, "spk_ids.json"), "w"))

        # train the model
        with open(self.config.finetune_config) as f:
            hparams = load_hyperpyyaml(
                f, overrides={"output_folder": dump_dir, "out_n_neurons": n_speakers}
            )
        train_data = prepare_dataset(hparams, train_datafile)
        speaker_brain = SpeakerBrain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
        )
        speaker_brain.fit(
            speaker_brain.hparams.epoch_counter,
            train_data,
            train_loader_kwargs=hparams["dataloader_options"],
        )

        # save the embedding model and load it
        checkpointer = Checkpointer(
            dump_dir,
            recoverables={
                "embedding_model": speaker_brain.modules.embedding_model,
                "classifier": speaker_brain.modules.classifier,
            },
        )
        checkpointer.save_checkpoint(name="spkid_model")
        shutil.copy(
            os.path.join(self.save_dir, "hyperparams.yaml"),
            os.path.join(dump_dir, "CKPT+spkid_model"),
        )
        self.model = EncoderClassifier.from_hparams(
            source=os.path.join(dump_dir, "CKPT+spkid_model"),
            run_opts={"device": self.device},
        )
