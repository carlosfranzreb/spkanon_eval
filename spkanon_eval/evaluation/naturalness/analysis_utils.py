"""Helper functions to analyze the results of the NISQA evaluation."""


import numpy as np


def analyse_func(indices, data):
    """
    - Filter the samples with the given indices.
    - Compute the average MOS score and return it as a string.
    """
    return [str(sum(indices)), str(np.round(np.mean(data[indices]), 2))]


def headers_func(dump_file, key=None):
    """
    Create the header of the dump file. If a key is given, write it to the right of the
    datafile.
    """
    with open(dump_file, "w") as f:
        if key is not None:
            f.write(f"dataset {key} n_samples mos\n")
        else:
            f.write("dataset n_samples mos\n")
