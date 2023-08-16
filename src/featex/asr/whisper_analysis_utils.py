"""Helper functions to analyze the results of the ASR experiments."""


import numpy as np


def analyse_func(indices, data):
    """
    - data is a tuple of the form (embs, dims)
    - Filter the samples with the given indices.
    - Compute the average embedding cosine similarity.
    - For each emotion dimension, compute the average difference between the 
        anonymized and the original audio, both in relative and absolute value.
    The output is a list of the form: [
        n_samples, avg_similarity, avg_arousal_diff, avg_dominance_diff,
        avg_valence_diff, avg_arousal_absdiff, avg_dominance_absdiff,
        avg_valence_absdiff
    ]
    """
    n_edits, n_words_ref = data
    avg_edits = np.round(np.mean(n_edits[indices]), 2)
    avg_words_ref = np.round(np.mean(n_words_ref[indices]), 2)
    avg_wer = np.round(avg_edits / avg_words_ref, 2)
    return [str(sum(indices)), str(avg_edits), str(avg_words_ref), str(avg_wer)]


def headers_func(dump_file, key=None):
    """
    Create the header of the dump file. If a key is given, write it to the right of the
    datafile.
    """
    with open(dump_file, "w") as f:
        if key is not None:
            f.write(f"dataset {key} n_samples avg_edits avg_words_ref wer\n")
        else:
            f.write("dataset n_samples avg_edits avg_words_ref wer\n")
