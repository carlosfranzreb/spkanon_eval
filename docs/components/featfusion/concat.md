# Feature Concatenation

The `Concat` class, defined in `src/featfusion/concat.py`, concatenates features from different sources into a single feature representation. It can be used to combine time-variant and time-invariant features, like speech and speaker representations.

The `config` dictionary should have two keys:

- `"time_dependent"`: A list of strings representing the names of the time-dependent features.
- `"time_invariant"`: A list of strings representing the names of the time-invariant features.

When given a batch, it takes a dictionary `feats` as input, where each key represents the feature name and the corresponding value is the feature tensor. It concatenates the features specified in the configuration (`self.timed` and `self.nontimed`) along the time dimension (if time-dependent) or as an additional channel (if time-invariant).
