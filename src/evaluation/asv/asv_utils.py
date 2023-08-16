import os
import json
import logging

import numpy as np
from sklearn.metrics import roc_curve, RocCurveDisplay
from matplotlib import pyplot as plt

from src.evaluation.analysis import get_characteristics


LOGGER = logging.getLogger("progress")


def compute_eer(sources, targets, llrs):
    """
    Compute the equal error rate (EER) for the given LLRs. The EER is the threshold
    that minimizes the absolute difference between the false acceptance rate (FAR)
    and the false rejection rate (FRR). We compute it with sklearn's roc_curve.

    Args:
    - pairs: sorted numpy array with all pairs of trial and enrollment utterances
    - llrs: numpy array containing the log-likelihood ratio (LLR) of each pair

    Returns:
    - fpr: numpy array containing the false positive rates (FPR) for each threshold
    - tpr: numpy array containing the true positive rates (TPR) for each threshold
    - thresholds: numpy array containing the thresholds
    - key: index of the threshold that is closest to the EER
    """
    # check that there are sources and targets
    if len(sources) == 0 or len(targets) == 0:
        LOGGER.warn("There are no sources or targets; cannot compute EER")
        return None, None, None, -1
    # compute the ROC curve
    same_speaker = sources == targets
    fpr, tpr, thresholds = roc_curve(same_speaker, llrs)
    # check that there are no NaNs
    if np.any(np.isnan(fpr)):
        LOGGER.warn("There are no different-speaker pairs; cannot compute EER")
        key = -1
    elif np.any(np.isnan(tpr)):
        LOGGER.warn("There are no same-speaker pairs; cannot compute EER")
        key = -1
    # compute the EER threshold
    else:
        key = np.nanargmin(np.absolute(((1 - tpr) - fpr)))
    return fpr, tpr, thresholds, key


def analyse_results(datafile, llr_file):
    """
    Compute and dump the EER and ROC curve for the whole dataset and each of its
    subsets w.r.t speaker characteristics present in the file (age, gender, etc.).
    """
    LOGGER.info(f"Analysing ASV results for {datafile}")
    # get the filename of the datafile
    fname = os.path.splitext(os.path.basename(datafile))[0]
    # get the dump folder for this datafile (the LLR file's folder)
    dump_folder = os.path.dirname(llr_file)
    # read the llr file and extract the source & target speakers and the llrs
    trials, enrolls, llrs = np.hsplit(np.load(llr_file), 3)
    spk_labels = np.load(os.path.join(dump_folder, "spk_labels.npy"))

    # compute the EER of the whole dataset and dump the ROC curve
    LOGGER.info("Computing EER of the whole dataset and dumping ROC curve")
    fpr, tpr, thresholds, key = compute_eer(trials, enrolls, llrs)
    # create the eer file with the headers if it doesn't exist yet
    eer_file = os.path.join(os.path.dirname(dump_folder), "eer.txt")
    if not os.path.exists(eer_file):
        with open(eer_file, "w") as f:
            f.write("dataset n_pairs threshold eer\n")
    # dump the EER and the threshold to the eer file
    with open(eer_file, "a") as f:
        f.write(f"{fname} {llrs.size} {thresholds[key]} {fpr[key]}\n")
    # dump the ROC curve
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.plot([0, 1], [0, 1], "k--")
    plt.savefig(os.path.join(dump_folder, "roc_curve.png"))

    # store the spk label for each value of each speaker char.
    speaker_chars, _, _ = get_characteristics(datafile, spk_labels)

    # for each speaker char., compute the EER for all its values
    for key, values in speaker_chars.items():
        # iterate over all combinations of values
        for value in values:
            LOGGER.info(f"Computing EER for {key} {value}")
            # get the indices of the rows that contain the values
            indices = np.where(
                np.logical_and(
                    np.isin(trials, values[value]), np.isin(enrolls, values[value])
                )
            )[0]
            # compute the EER and dump it to the dump file if it was computed
            fpr, tpr, thresholds, eer_key = compute_eer(
                trials[indices], enrolls[indices], llrs[indices]
            )
            if eer_key >= 0:
                dump_file = os.path.join(os.path.dirname(dump_folder), f"eer_{key}.txt")
                if not os.path.exists(dump_file):
                    with open(dump_file, "w") as f:
                        f.write(f"dataset {key} n_pairs threshold eer\n")
                with open(dump_file, "a") as f:
                    f.write(f"{fname} {value} {indices.size} ")
                    f.write(f"{thresholds[eer_key]} {fpr[eer_key]}\n")


def count_speakers(datafile):
    """Count the number of speakers in the datafile."""
    with open(datafile) as f:
        current_spk = None
        n_speakers = 0
        for line in f:
            obj = json.loads(line.strip())
            if obj["label"] != current_spk:
                current_spk = obj["label"]
                n_speakers += 1
    LOGGER.info(f"Number of speakers in {datafile}: {n_speakers}")
    return n_speakers


def filter_samples(datafile, filter_cfg):
    """
    Filter out samples that do not meet the criteria given in the config. The
    criteria may define the minimum duration of the samples and the minimum number
    of samples per speaker.
    Store the results in the experiment folder and return the path to the
    filtered datafile. This method assumes that the samples of each speaker
    are all placed together in the datafile.
    Also, return the number of speakers that remain after the filtering.
    """
    LOGGER.info(f"Filtering samples in {datafile}")
    min_samples = filter_cfg.get("min_samples", 1)
    min_duration = filter_cfg.get("min_dur", 0)
    folder, file = os.path.split(datafile)
    filename, ext = os.path.splitext(file)
    os.makedirs(os.path.join(folder, "asv_splits"), exist_ok=True)
    out_file = os.path.join(folder, "asv_splits", f"{filename}_filtered{ext}")
    writer = open(out_file, "w")
    n_speakers = 0
    with open(datafile) as f:
        spk_lines = list()
        current_spk = None
        for line in f:
            obj = json.loads(line.strip())
            if obj["label"] != current_spk:
                if len(spk_lines) >= min_samples:
                    writer.writelines(spk_lines)
                    n_speakers += 1
                current_spk = obj["label"]
                spk_lines = list()
            if obj["duration"] >= min_duration:
                spk_lines.append(line)
        if len(spk_lines) >= min_samples:
            writer.writelines(spk_lines)
            n_speakers += 1
    writer.close()
    LOGGER.info(f"Filtered datafile saved to {out_file}")
    return out_file, n_speakers


def compute_llrs(plda, vecs, chunk_size):
    """
    Compute the log-likelihood ratios (LLRs) of all pairs of trial and enrollment
    utterances. For each speaker, the first utterance is considered the trial
    utterance and the rest are considered the enrollment utterances.
    The LLRs are calculated as in the plda package, but the code is adapted to our
    use case, where every vector is used multiple times. We therefore compute the
    marginal LLs beforehand once, and use them to compute the LLRs for all pairs.

    Args:
    - vecs: dict with two keys, trials and enrolls, each containing a numpy array
        containing speaker embeddings.

    Returns:
    - llrs: numpy array with the LLRs
    - indices: numpy array containing the indices of trial and enroll utterances
        that were used to compute each LLR. The number of pairs equals all possible
        combinations of trial and enroll utts.
    """
    LOGGER.info("Computing LLRs for all pairs of trial and enrollment utterances")
    # compute all pairs of trial and enrollment utterances
    indices = np.dstack(
        np.meshgrid(
            np.arange(vecs["trials"].shape[0]), np.arange(vecs["enrolls"].shape[0])
        )
    ).reshape(-1, 2)

    # iterate over chunks of `chunk_size` pairs to avoid memory issues
    llrs = None
    for i in range(0, indices.shape[0], chunk_size):
        # define the indices and vectors of the current chunk
        idx = indices[i : i + chunk_size]
        data = {"trials": dict(), "enrolls": dict()}
        for i, key in enumerate(data):
            data[key]["idx"] = idx[:, i]
            data[key]["idx_unique"], data[key]["idx_inverse"] = np.unique(
                data[key]["idx"], return_inverse=True
            )
            data[key]["vecs"] = vecs[key][data[key]["idx_unique"]]
            # add a new dimension to the vectors to conform to PLDA's input format
            data[key]["vecs"] = data[key]["vecs"][:, np.newaxis, :]
            # compute the marginal log-likelihoods of the vectors
            data[key]["lls"] = plda.model.calc_logp_marginal_likelihood(
                data[key]["vecs"]
            )
            # map vecs and lls back to the original indices
            data[key]["vecs"] = data[key]["vecs"][data[key]["idx_inverse"]]
            data[key]["lls"] = data[key]["lls"][data[key]["idx_inverse"]]
        # compute the LLRs for the current chunk
        pairs = np.concatenate(
            [data["trials"]["vecs"], data["enrolls"]["vecs"]], axis=1,
        )
        pair_lls = plda.model.calc_logp_marginal_likelihood(pairs)
        chunk_llrs = pair_lls - (data["trials"]["lls"] + data["enrolls"]["lls"])
        # add the LLRs of the current chunk to the total LLRs
        if llrs is None:
            llrs = chunk_llrs
        else:
            llrs = np.concatenate((llrs, chunk_llrs))
    return llrs, indices
