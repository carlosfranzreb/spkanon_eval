"""
Slightly simplified version of the ASV system from the VPC 2022.

### Training phase

This phase requires a dataset different from the one used for evaluation.

1. Remove utterances that are too short, and speakers with too few utterances.
2. Fine-tune NeMo's SpkId net.
3. Extract SpkId vectors from fine-tuned net.
4. Center the SpkId vectors.
5. Decrease the dimensionality of the centered SpkId vectors with LDA.
6. Train the PLDA model with the vectors resulting from LDA.

### Evaluation phase

1. Split the evaluation data into trial and enrollment data.
2. Use anonymized utterances of trial data, and maybe anonymized enrollment data.
3. Compute the SpkId vectors of all trial and enrollment utterances.
4. Compute LLRs of all pairs of trial and enrollment utterances with the trained PLDA
    model.
5. Average the LLRS across speakers.
6. Compute the ROC curve and the EER.

### Two attack scenarios

1. Ignorant: the attacker does not have access to the anonymization model. The training
    data of the ASV system and the enrollment data are not anonymized.
2. Lazy-informed: the attacker has access to the anonymization model. The training data
    of the ASV system is anonymized without consistent targets, and the enrollment data
    is anonymized with consistent targets.
"""


import os
import pickle
import logging
import copy

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plda

from src.setup_module import setup
from src.dataloader import setup_dataloader, collate_fn_inference
from src.inference import infer
from src.evaluation.asv.asv_utils import (
    analyse_results,
    filter_samples,
    count_speakers,
    compute_llrs,
)
from src.evaluation.asv.trials_enrolls import split_trials_enrolls


SAMPLE_RATE = 16000  # sample rate of the spkid model
LOGGER = logging.getLogger("progress")
CHUNK_SIZE = 5000  # defines how many LLRs are computed in parallel


class ASV:
    def __init__(self, config, device, model=None):
        """Initialize the ASV system."""

        # delete the model if it is not used
        self.scenario = config.scenario
        if config.scenario == "lazy-informed":
            self.model = model
            if self.model is None:
                raise ValueError("Lazy-informed scenario requires a model")
        else:
            self.model = None

        self.device = device
        self.config = config

        # prepare the spkid config and init the model
        self.spkid_model = setup(config.spkid, device)

        # disable augmentation if defined in config
        if "augmentor" in self.config.data.config:
            self.config.data.config.augmentor = None

        # init the LDA model with ckpt if possible, else declare it null
        lda_ckpt = config.get("lda_ckpt", None)
        self.lda_model = None
        if lda_ckpt is not None:
            LOGGER.info(f"Loading LDA ckpt `{lda_ckpt}`")
            self.lda_model = pickle.load(open(lda_ckpt, "rb"))

        # init the PLDA model with ckpt if possible, else declare it null
        plda_ckpt = config.get("plda_ckpt", None)
        self.plda_model = None
        if plda_ckpt is not None:
            LOGGER.info(f"Loading PLDA ckpt `{plda_ckpt}`")
            self.plda_model = pickle.load(open(plda_ckpt, "rb"))

    def train(self, exp_folder, datafiles):
        """
        Train the PLDA model with the SpkId vectors and also the SpkId model.
        The anonymized samples are stored in the given path `exp_folder`.
        If the scenario is "lazy-informed", the training data is anonymized without
        consistent targets.
        """

        # define and create the directory where models and training data are stored
        dump_dir = os.path.join(exp_folder, "eval", "asv-plda", self.scenario, "train")
        os.makedirs(dump_dir, exist_ok=True)

        # If the scenario is "lazy-informed", anonymize the training data
        if self.config.scenario == "lazy-informed":
            LOGGER.info(f"Anonymizing training data: {datafiles}")
            datafiles = self.anonymize_data(exp_folder, datafiles, dump_dir, False)

        # filter samples if necessary and count speakers
        train_files = list()
        n_speakers = 0
        for datafile in datafiles:
            if "filter" in self.config:  # filter files if defined in config
                datafile, n_spk = filter_samples(datafile, self.config.filter)
                n_speakers += n_spk
            else:  # otherwise, count the number of speakers
                n_speakers += count_speakers(datafile)
            train_files.append(datafile)
        LOGGER.info(f"Number of speakers in training files: {n_speakers}")

        # fine-tune SpkId model and store the ckpt if needed
        if self.config.spkid.train:
            self.spkid_model.finetune(exp_folder, train_files, n_speakers)

        # compute SpkId vectors of all utterances with fine-tuned net and center them
        vecs, labels = self.compute_spkid_vecs(train_files)
        vecs -= np.mean(vecs, axis=0)

        # create the directory where models are stored
        models_dir = os.path.join(dump_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # perform dimensionality reduction with LDA if possible and store the model
        reduced_dims = self.config.get("reduced_dims", None)
        if reduced_dims is not None and reduced_dims <= min(vecs.shape[1], n_speakers):
            LOGGER.info(f"Reducing dimensionality to {reduced_dims} with LDA")
            self.lda_model = LinearDiscriminantAnalysis(n_components=reduced_dims)
            vecs = self.lda_model.fit_transform(vecs, labels)
            pickle.dump(
                self.lda_model, open(os.path.join(models_dir, "lda.pkl"), "wb"),
            )

        # train the PLDA model and store the model
        LOGGER.info("Training PLDA model")
        self.plda_model = plda.Classifier()
        self.plda_model.fit_model(vecs, labels)
        if self.plda_model.model.pca is not None:
            n_components = self.plda_model.model.pca.components_.shape[0]
            LOGGER.warn(f"PCA is used within PLDA with {n_components} components")
        pickle.dump(self.plda_model, open(os.path.join(models_dir, "plda.pkl"), "wb"))

    def eval_dir(self, exp_folder, datafiles, is_baseline):
        """
        Evaluate the ASV system on the given directory. The results of the
        anonymization are stored in the given path `exp_folder`, in the folder
        `data`. The results of this evaluation are stored in `eval`.
        The first utterance of each speaker is considered the trial utterances and
        the rest are considered the enrollment utterances.
        """
        dump_folder = os.path.join(exp_folder, "eval", "asv-plda", self.scenario)

        # iterate over the evaluation files and evaluate them separately
        for datafile in datafiles:
            fname = os.path.splitext(os.path.basename(datafile))[0]
            LOGGER.info(f"ASV evaluation of datafile `{fname}`")
            dump_subfolder = os.path.join(dump_folder, "results", fname)
            os.makedirs(dump_subfolder, exist_ok=True)

            # split the datafile into trial and enrollment datafiles
            f_trials, f_enrolls = split_trials_enrolls(
                exp_folder,
                datafile.replace("results/", ""),
                self.config.data.config.root_folder,
                is_baseline,
            )

            # if the f_trials or f_enrolls do not exist, skip the evaluation
            if not (os.path.exists(f_trials) and os.path.exists(f_enrolls)):
                LOGGER.warning("No trials to evaluate; skipping")
                continue

            # If the scenario is "lazy-informed", anonymize the enrollment data
            if self.config.scenario == "lazy-informed":
                LOGGER.info("Anonymizing enrollment data of the ASV system")
                f_enrolls = self.anonymize_data(
                    exp_folder,
                    [f_enrolls],
                    os.path.join(dump_folder, "anon-enrolls"),
                    True,
                )[0]

            # compute SpkId vectors of all utts and map them to PLDA space
            vecs, labels = dict(), dict()
            for name, f in zip(["trials", "enrolls"], [f_trials, f_enrolls]):
                vecs[name], labels[name] = self.compute_spkid_vecs(f)
                vecs[name] -= np.mean(vecs[name], axis=0)
                if self.lda_model is not None:
                    vecs[name] = self.lda_model.transform(vecs[name])
                vecs[name] = self.plda_model.model.transform(
                    vecs[name], from_space="D", to_space="U_model"
                )

            # dump labels & replace trial and enrollment labels with indices
            spk_labels = np.copy(labels["trials"])
            np.save(os.path.join(dump_subfolder, "spk_labels.npy"), spk_labels)
            labels["trials"] = np.arange(len(labels["trials"]))
            labels["enrolls"] = np.array(
                [labels["trials"][spk_labels == l][0] for l in labels["enrolls"]]
            )
            del spk_labels

            # compute LLRs of all pairs of trial and enrollment utterances
            llrs, pairs = compute_llrs(self.plda_model, vecs, CHUNK_SIZE)
            del vecs

            # map utt indices to speaker indices
            pairs[:, 0] = labels["trials"][pairs[:, 0]]
            pairs[:, 1] = labels["enrolls"][pairs[:, 1]]

            # avg. LLRs across speakers and dump them to the experiment folder
            LOGGER.info("Averaging LLRs across speakers")
            LOGGER.info(f"No. of speaker pairs: {pairs.shape[0]}")
            llr_file = os.path.join(dump_subfolder, "llrs.npy")
            unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
            llr_avgs = np.bincount(inverse, weights=llrs) / np.bincount(inverse)
            np.save(llr_file, np.hstack((unique_pairs, llr_avgs.reshape(-1, 1))))

            # compute the EER for the data and its subsets w.r.t. speaker chars.
            analyse_results(datafile, llr_file)

    def anonymize_data(self, exp_folder, datafiles, dump_dir, consistent_targets):
        """
        Anonymize the given datafiles with the `src.inference.infer` function. The
        anonymized audiofiles and the corresponding datafiles are stored in the given
        directory `dump_dir`. `consistent_targets` indicates whether the targets
        selected for the source speakers are consistent across their utterances or not.
        This method returns the paths to the anonymized datafiles.
        """
        self.model.set_consistent_targets(consistent_targets)
        infer(
            self.model,
            exp_folder,
            datafiles,
            dump_dir,
            self.config.inference.input,
            self.config.data.config,
            self.config.sample_rate,
        )
        return [f.replace(exp_folder, dump_dir) for f in datafiles]

    def compute_spkid_vecs(self, datafiles):
        """
        Compute the SpkId vectors of all speakers and return them along with their
        speaker labels. The labels returned here are not the same as those in the
        datafiles, because the NeMo dataloader changes them so that they are
        consecutive integers starting from 0. We need to map them back to the original
        labels. We do so with NeMo's id2label attribute.
        """
        LOGGER.info(f"Computing SpkId vectors of files {datafiles}")
        labels = np.array([], dtype=int)  # utterance labels
        vecs = None  # spkid vecs of utterances

        spkid_config = copy.deepcopy(self.config.data.config)
        spkid_config.trainer = self.config.spkid.trainer
        spkid_config.sample_rate = SAMPLE_RATE
        dl = setup_dataloader(spkid_config, datafiles, collate_fn=collate_fn_inference)
        for batch in dl:
            new_vecs = self.spkid_model.run(batch).detach().cpu().numpy()
            vecs = new_vecs if vecs is None else np.vstack([vecs, new_vecs])
            new_labels = np.array(
                [dl.dataset.id2label[l] for l in batch[2].detach().cpu().numpy()]
            )
            labels = np.concatenate([labels, new_labels])
        return vecs, labels
