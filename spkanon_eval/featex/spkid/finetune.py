#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""


import random
import torch
import torchaudio
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset


class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"""

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        try:
            batch = batch.to(self.device)
            wavs, lens = batch.sig

            if stage == sb.Stage.TRAIN:
                wavs_aug_tot = []
                wavs_aug_tot.append(wavs)
                for augment in self.hparams.augment_pipeline:
                    wavs_aug = augment(wavs, lens)

                    if wavs_aug.shape[1] > wavs.shape[1]:
                        wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                    else:
                        zero_sig = torch.zeros_like(wavs)
                        zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                        wavs_aug = zero_sig

                    if self.hparams.concat_augment:
                        wavs_aug_tot.append(wavs_aug)
                    else:
                        wavs = wavs_aug
                        wavs_aug_tot[0] = wavs

                wavs = torch.cat(wavs_aug_tot, dim=0)
                self.n_augment = len(wavs_aug_tot)
                lens = torch.cat([lens] * self.n_augment)

            feats = self.modules.compute_features(wavs)
            feats = self.modules.mean_var_norm(feats, lens)
            embeddings = self.modules.embedding_model(feats)
            outputs = self.modules.classifier(embeddings)

            return outputs, lens
        except Exception as e:
            print("Error in compute_objectives")
            print("predictions:", predictions.shape)
            print("batch:", batch.shape)
            print(e)
            import sys
            sys.exit()

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        try:
            predictions, lens = predictions
            predictions = predictions.to(self.device)
            lens = lens.to(self.device)
            spkid = torch.tensor([int(spkid) for spkid in batch.spk_id_encoded]).unsqueeze(
                1
            ).to(self.device)

            if stage == sb.Stage.TRAIN:
                spkid = torch.cat([spkid] * self.n_augment, dim=0).to(self.device)

            loss = self.hparams.compute_cost(predictions, spkid, lens)
            if stage == sb.Stage.TRAIN and hasattr(
                self.hparams.lr_annealing, "on_batch_end"
            ):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

            return loss
        except Exception as e:
            print("Error in compute_objectives")
            print("predictions:", predictions.shape)
            print("batch:", batch.shape)
            print(e)
            import sys
            sys.exit()


    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage == sb.Stage.VALID:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            if epoch > 0:
                old_lr, new_lr = self.hparams.lr_annealing(epoch)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
                train_stats = self.train_stats
            else:
                old_lr = 0.0
                train_stats = None
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=train_stats,
                valid_stats=stage_stats,
            )


def prepare_dataset(hparams: dict, datafile: str) -> DynamicItemDataset:
    "Creates the datasets and their data processing pipelines."

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])
    data = DynamicItemDataset.from_csv(
        csv_path=datafile,
    )
    datasets = [data]

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, duration):
        duration_sample = int(duration * hparams["sample_rate"])
        if hparams["random_chunk"] and duration_sample > snt_len_sample:
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
            start = 0
            stop = duration_sample
        num_frames = stop - start
        sig, fs = torchaudio.load(wav, num_frames=num_frames, frame_offset=start)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return data
