"""Helper functions to analyze the results of the SER experiments."""


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
    embs, dims = data
    results = [str(sum(indices)), str(np.round(np.mean(embs[indices]), 2))]
    for dim in dims:
        results.append(str(np.round(np.mean(dims[dim][indices]), 2)))
        results.append(str(np.round(np.mean(np.abs(dims[dim][indices])), 2)))
    return results


def headers_func(dump_file, key=None):
    """
    Create the header of the dump file. If a key is given, write it to the right of the
    datafile.
    """
    dims = ["arousal", "dominance", "valence"]
    with open(dump_file, "w") as f:
        if key is not None:
            f.write(f"dataset {key} n_samples similarity ")
        else:
            f.write("dataset n_samples similarity ")
        f.write("".join([f"{dim}_diff " for dim in dims]))
        f.write(" ".join([f"{dim}_absdiff" for dim in dims]) + "\n")
