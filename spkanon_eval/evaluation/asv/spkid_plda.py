"""
Slightly simplified version of the ASV system from the VPC 2022.

### Training phase

This phase requires a dataset different from the one used for evaluation.

1. Remove utterances that are too short, and speakers with too few utterances.
2. Fine-tune SpeechBrain's SpkId net.
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

from spkanon_eval.setup_module import setup
from spkanon_eval.datamodules.dataloader import setup_dataloader
from spkanon_eval.inference import infer
from spkanon_eval.evaluation.asv.asv_utils import (
    analyse_results,
    count_speakers,
    compute_llrs,
)
from spkanon_eval.evaluation.asv.trials_enrolls import split_trials_enrolls
from spkanon_eval.component_definitions import EvalComponent


SAMPLE_RATE = 16000  # sample rate of the spkid model
LOGGER = logging.getLogger("progress")
CHUNK_SIZE = 5000  # defines how many LLRs are computed in parallel


class ASV(EvalComponent):
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

    def train(self, exp_folder: str) -> None:
        """
        Train the PLDA model with the SpkId vectors and also the SpkId model.
        The anonymized samples are stored in the given path `exp_folder`.
        If the scenario is "lazy-informed", the training data is anonymized without
        consistent targets.
        """

        datafile = os.path.join(exp_folder, "data", "train_eval.txt")

        # define and create the directory where models and training data are stored
        dump_dir = os.path.join(exp_folder, "eval", "asv-plda", self.scenario, "train")
        os.makedirs(dump_dir, exist_ok=True)

        # If the scenario is "lazy-informed", anonymize the training data
        if self.config.scenario == "lazy-informed":
            LOGGER.info(f"Anonymizing training data: {datafile}")
            datafile = self.anonymize_data(exp_folder, "train_eval", False)

        n_speakers = count_speakers(datafile)
        LOGGER.info(f"Number of speakers in training file: {n_speakers}")

        # fine-tune SpkId model and store the ckpt if needed
        if self.config.spkid.train:
            self.spkid_model.finetune(
                os.path.join(dump_dir, "spkid"), datafile, n_speakers
            )

        # compute SpkId vectors of all utterances with spkid model and center them
        vecs, labels = self.compute_spkid_vecs(datafile)
        train_mean_spkemb = np.mean(vecs, axis=0)
        vecs -= train_mean_spkemb

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
                self.lda_model,
                open(os.path.join(models_dir, "lda.pkl"), "wb"),
            )

        # train the PLDA model and store the model
        LOGGER.info("Training PLDA model")
        self.plda_model = plda.Classifier()
        self.plda_model.train_mean_spkemb = train_mean_spkemb
        self.plda_model.fit_model(vecs, labels)
        if self.plda_model.model.pca is not None:
            n_components = self.plda_model.model.pca.components_.shape[0]
            LOGGER.warn(f"PCA is used within PLDA with {n_components} components")
        pickle.dump(self.plda_model, open(os.path.join(models_dir, "plda.pkl"), "wb"))

    def eval_dir(self, exp_folder: str, datafile: str, is_baseline: bool) -> None:
        """
        Evaluate the ASV system on the given directory. The results of the
        anonymization are stored in the given path `exp_folder`, in the folder
        `data`. The results of this evaluation are stored in `eval`.
        The first utterance of each speaker is considered the trial utterances and
        the rest are considered the enrollment utterances.

        Args:
            exp_folder: path to the experiment folder
            datafile: datafile to evaluate
            is_baseline: whether the baseline is being evaluated
        """
        dump_folder = os.path.join(exp_folder, "eval", "asv-plda", self.scenario)

        fname = os.path.splitext(os.path.basename(datafile))[0]
        dump_subfolder = os.path.join(dump_folder, "results", fname)
        os.makedirs(dump_subfolder, exist_ok=True)

        # split the datafile into trial and enrollment datafiles
        root_dir = None if is_baseline else self.config.data.config.root_folder
        anon_folder = self.config.data.config.get("anon_folder", None)
        f_trials, f_enrolls = split_trials_enrolls(exp_folder, root_dir, anon_folder)

        # if the f_trials or f_enrolls do not exist, skip the evaluation
        if not (os.path.exists(f_trials) and os.path.exists(f_enrolls)):
            LOGGER.warning("No trials to evaluate; stopping component evaluation")
            return

        # If the scenario is "lazy-informed", anonymize the enrollment data
        if self.config.scenario == "lazy-informed":
            LOGGER.info("Anonymizing enrollment data of the ASV system")
            f_enrolls = self.anonymize_data(exp_folder, "eval_enrolls", True)

        # compute SpkId vectors of all utts and map them to PLDA space
        vecs, labels = dict(), dict()
        for name, f in zip(["trials", "enrolls"], [f_trials, f_enrolls]):
            vecs[name], labels[name] = self.compute_spkid_vecs(f)
            vecs[name] -= self.plda_model.train_mean_spkemb
            if self.lda_model is not None:
                vecs[name] = self.lda_model.transform(vecs[name])
            vecs[name] = self.plda_model.model.transform(
                vecs[name], from_space="D", to_space="U_model"
            )

        # compute LLRs of all pairs of trial and enrollment utterances
        llrs, pairs = compute_llrs(self.plda_model, vecs, CHUNK_SIZE)
        del vecs

        # map utt indices to speaker indices
        pairs[:, 0] = labels["trials"][pairs[:, 0]]
        pairs[:, 1] = labels["enrolls"][pairs[:, 1]]

        # avg. LLRs across speakers and dump them to the experiment folder
        LOGGER.info("Averaging LLRs across speakers")
        LOGGER.info(f"No. of speaker pairs: {pairs.shape[0]}")
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        llr_avgs = np.bincount(inverse, weights=llrs) / np.bincount(inverse)

        llr_file = os.path.join(dump_subfolder, "llrs.npy")
        np.save(llr_file, np.hstack((unique_pairs, llr_avgs.reshape(-1, 1))))

        # compute the EER for the data and its subsets w.r.t. speaker chars.
        analyse_results(datafile, llr_file)

    def anonymize_data(
        self, exp_folder: str, df_name: str, consistent_targets: bool
    ) -> str:
        """
        Anonymize the given datafile and return the path to the anonymized datafile.

        Args:
            exp_folder: path to the experiment folder
            df_name: name of the datafile, without the directory or the extension
                (e.g. "f_enrolls"). The corresponding datafile is assumed to be in
                `{exp_folder}/data`.
            consistent_targets: whether each speaker should always be anonymized with
            the same target.
        """
        self.model.set_consistent_targets(consistent_targets)
        anon_datafile = infer(exp_folder, df_name, self.model, self.config)
        return anon_datafile

    def compute_spkid_vecs(self, datafile: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the SpkId vectors of all speakers and return them along with their
        speaker labels. The labels returned here are not the same as those in the
        datafiles, because our dataset changes them so that they are consecutive
        integers starting from 0. You can find the mapping in the experiment folder,
        under `spk2id.json`.

        Args:
            datafile: path to the datafile

        Returns:
            SpkId vectors: (n_utterances, embedding_dim)
            speaker labels: (n_utterances,)
        """
        LOGGER.info(f"Computing SpkId vectors of {datafile}")
        labels = np.array([], dtype=int)  # utterance labels
        vecs = None  # spkid vecs of utterances

        spkid_config = copy.deepcopy(self.config.data.config)
        spkid_config.batch_size = self.config.spkid.batch_size
        spkid_config.sample_rate = SAMPLE_RATE
        dl = setup_dataloader(spkid_config, datafile)
        for batch in dl:
            new_vecs = self.spkid_model.run(batch).detach().cpu().numpy()
            vecs = new_vecs if vecs is None else np.vstack([vecs, new_vecs])
            new_labels = batch[1].detach().cpu().numpy()
            labels = np.concatenate([labels, new_labels])
        return vecs, labels
