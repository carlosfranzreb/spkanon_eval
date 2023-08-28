"""
Helper functions related to splitting the data into trial and enrollment utterances.
"""


import os
import json
import logging


LOGGER = logging.getLogger("progress")


def split_trials_enrolls(exp_folder, datafile, root_folder, is_baseline):
    """
    Split the datafile into trial and enrollment datafiles.

    - The first utterance of each speaker is considered the trial utterance and the
        rest are considered enrollment utterances.
    - If there is no anonymized path for the first utterance, it is added to the
        enroll utts. The next utt is considered the trial utt.
    - If all the utts of a speaker have been read and either the trial or enroll
        utt is still missing, the speaker is not added to the datafiles.
    - The new files are stored in the experiment folder and their paths returned.
    """

    LOGGER.info(f"Splitting datafile {datafile} into trial and enrollment data")

    # create trial and enrollment files for this datafile
    folder, file = os.path.split(datafile)
    fname, ext = os.path.splitext(file)
    os.makedirs(os.path.join(folder, "asv_splits"), exist_ok=True)
    f_trials = os.path.join(folder, "asv_splits", f"{fname}_trials{ext}")
    f_enrolls = os.path.join(folder, "asv_splits", f"{fname}_enrolls{ext}")
    LOGGER.info(f"Trial data will be stored in {f_trials}")
    LOGGER.info(f"Enrollment data will be stored in {f_enrolls}")

    # if the files already exist, return them; if one of them only exists, delete it
    if os.path.exists(f_trials) and os.path.exists(f_enrolls):
        LOGGER.info("Trial and enrollment data already exist")
        return f_trials, f_enrolls
    elif os.path.exists(f_trials):
        LOGGER.info("Trial data already exists; removing it")
        os.remove(f_trials)
    elif os.path.exists(f_enrolls):
        LOGGER.info("Enrollment data already exists; removing it")
        os.remove(f_enrolls)

    # define variables to store the current speaker and its trial and enroll utts
    current_spk = None
    spk_trial = None
    spk_enrolls = list()

    # iterate over the samples of the datafile
    for line in open(datafile):
        obj = json.loads(line.strip())
        # object belongs to a new speaker; write the previous speaker
        if obj["label"] != current_spk:
            # if this is the first speaker, set the current speaker
            if current_spk is None:
                current_spk = obj["label"]
            else:
                # write the previous spk if it has trial and enroll utts
                write_speaker(current_spk, spk_trial, spk_enrolls, f_trials, f_enrolls)
                # reset the speaker variables
                spk_trial = None
                spk_enrolls = list()
                current_spk = obj["label"]

        # if there is no trial utterance for this speaker, set it
        if spk_trial is None:
            anon_obj = set_anon_path(exp_folder, obj, root_folder, is_baseline)
            if anon_obj is not None:
                spk_trial = anon_obj

        # otherwise add the original path to enrolls
        else:
            spk_enrolls.append(obj)

    # write the last speaker
    write_speaker(current_spk, spk_trial, spk_enrolls, f_trials, f_enrolls)
    return f_trials, f_enrolls


def set_anon_path(exp_folder, obj, root_folder, is_baseline):
    """
    Modify the path of the utterance so that it references the anonymized
    utterance, which is stored in the experiment folder under `results`.
    - Ensure that the anonymized path exists. If not, log a warning and return
        null.
    - If we are evaluating the baseline, return the original object.
    """
    if is_baseline is True:
        return obj
    rel_path = obj["audio_filepath"][len(root_folder) + 1 :]
    obj["audio_filepath"] = os.path.join(exp_folder, "results", rel_path)
    if not os.path.exists(obj["audio_filepath"]):
        LOGGER.warn(f"File '{obj['audio_filepath']}' does not exist")
        return None
    return obj


def write_speaker(spk, trial_data, enroll_data, trial_file, enroll_file):
    """
    Write the trial and enrollment data to their corresponding files if possible.
    If either the trial or an enrollment utterance is missing, the speaker is not
    included in the datafiles.
    - trial_data should be either a string containing a json object or None
    - enroll_data should be a list of strings containing json objects
    """
    if trial_data is not None and len(enroll_data) > 0:
        with open(trial_file, "a") as f:
            f.write(json.dumps(trial_data) + "\n")
        with open(enroll_file, "a") as f:
            for enroll_utt in enroll_data:
                f.write(json.dumps(enroll_utt) + "\n")
    elif trial_data is None:
        LOGGER.warn(f"No trial utterance found for speaker {spk}")
    elif len(enroll_data) == 0:
        LOGGER.warn(f"No enrollment utterances found for speaker {spk}")
