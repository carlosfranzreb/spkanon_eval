"""
Helper functions related to splitting the data into trial and enrollment utterances.
"""

import os
import json
import logging


LOGGER = logging.getLogger("progress")


def split_trials_enrolls(
    exp_folder: str,
    root_folder: str = None,
    anon_folder: str = None,
    enrolls: list = None,
) -> tuple[str, str]:
    """
    Split the evaluation data into trial and enrollment datafiles. The first utt of
    each speaker is the trial utt, and the rest are enrollment utts. If the root folder
    is given, it is replaced in the trial with the folder where the anonymized
    evaluation data is stored (`exp_folder/results/anon_eval`). The root folder is None
    if we are evaluating the baseline, where speech is not anonymized.

    Args:
        exp_folder: path to the experiment folder.
        root_folder (optional): root folder of the data.
        anon_folder (optional): folder where the anonymized evaluation data is stored.
            It it is not given, we assume that it is the same as the experiment folder.
        enrolls: list of files defining the enrollment data. Each of these files
            contains one filename per line.

    Returns:
        paths to the created trial and enrollment datafiles

    Raises:
        ValueError: if one of the speakers only has one utterance. Each speaker should
            have at least two utterances, one for trial and one for enrollment.
    """

    LOGGER.info("Splitting evaluation data into trial and enrollment data")
    datafile = os.path.join(exp_folder, "data", "eval.txt")
    f_trials = os.path.join(exp_folder, "data", "eval_trials.txt")
    f_enrolls = os.path.join(exp_folder, "data", "eval_enrolls.txt")

    if os.path.exists(f_trials):
        LOGGER.warning("Datafile splits into trial and enrolls already exist, skipping")
        return f_trials, f_enrolls

    if root_folder is None:
        LOGGER.info("No root folder given: original trial data will be used.")

    if anon_folder is None:
        anon_folder = exp_folder

    # if enrolls are given, use them to split the data
    if enrolls is not None:
        trial_writer = open(f_trials, "w")
        enroll_writer = open(f_enrolls, "w")
        enroll_fnames = list()

        # gather the filenames of the enrollment data
        for enroll_file in enrolls:
            with open(enroll_file) as f:
                for line in f:
                    enroll_fnames.append(line.strip())

        # split the data into trial and enrollment data
        for line in open(datafile):
            obj = json.loads(line.strip())
            fname = os.path.splitext(os.path.basename(obj["path"]))[0]
            if fname in enroll_fnames:
                enroll_writer.write(line)
            else:
                obj["path"] = anon_path(obj["path"], anon_folder, root_folder)
                trial_writer.write(json.dumps(obj) + "\n")
        
        trial_writer.close()
        enroll_writer.close()

    # if no enrolls, the trial is the first utt of each speaker
    else:
        current_spk = None
        spk_objs = list()

        for line in open(datafile):
            obj = json.loads(line.strip())
            spk = obj["speaker_id"]
            if current_spk is None:
                current_spk = spk
            elif spk != current_spk:
                split_speaker(spk_objs, f_trials, f_enrolls, anon_folder, root_folder)
                spk_objs = list()
                current_spk = spk
            spk_objs.append(obj)

        split_speaker(spk_objs, f_trials, f_enrolls, anon_folder, root_folder)

    return f_trials, f_enrolls


def split_speaker(
    spk_data: list[dict],
    trial_file: str,
    enroll_file: str,
    exp_folder: str,
    root_folder: str = None,
) -> None:
    """
    Split the speaker's data into trial and enrollment data. The first utt is the trial
    utt, and the rest are enrollment utts. If the root folder is given, it is replaced
    in the trial with the folder where the anonymized evaluation data is stored
    (`exp_folder/results/eval`).

    Args:
        spk_data: list of datafile objects from one speaker.
        trial_file: path to the trial datafile.
        enroll_file: path to the enrollment datafile.
        exp_folder: path to the experiment folder.
        root_folder (optional): root folder of the data.

    Raises:
        ValueError: if one of the speakers only has one utterance. Each speaker should
            have at least two utterances, one for trial and one for enrollment.
    """

    if len(spk_data) == 1:
        error_msg = f"Speaker {spk_data[0]['speaker_id']} has only one utterance"
        LOGGER.error(error_msg)
        raise ValueError(error_msg)

    trial_sample, enroll_data = spk_data[0], spk_data[1:]
    if root_folder is not None:
        trial_sample["path"] = anon_path(trial_sample["path"], exp_folder, root_folder)

    with open(trial_file, "a") as f:
        f.write(json.dumps(trial_sample) + "\n")
    with open(enroll_file, "a") as f:
        for enroll_utt in enroll_data:
            f.write(json.dumps(enroll_utt) + "\n")


def anon_path(path: str, exp_folder: str, root_folder: str = None) -> str:
    """
    Replace the root folder in the path with the folder where the anonymized evaluation
    data is stored (`exp_folder/results/eval`).

    Args:
        path: the path to the file.
        exp_folder: path to the experiment folder.
        root_folder: root folder of the data.

    Returns:
        the path with the root folder replaced.
    """
    return path.replace(root_folder, os.path.join(exp_folder, "results", "eval"))
