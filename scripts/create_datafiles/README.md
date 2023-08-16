# Datafile creation

When creating a datafile, you must ensure that it satisfies the following requirements:

Datafiles are text files with one sample per line, in the form of a JSON object, which you can obtain by calling `json.dumps`. Each object must contain at least the following fields, with these keys:

`audio_filepath`: path to the audio file, where the path to the folder, after which the folder structure is the same, is replaced with `{root}`. This approach allows us to use the same datafile across machines.
`label`: the speaker ID, which we use for ASV evaluation and to analyse all results w.r.t. speaker metadata (age, gender, etc.)
`text`: the transcript of the utterance; relevant for ASR evaluation.
`duration`: duration of the clip, which we use to filter utterances that are either too short or too long, according to the experiment configuration.

All other fields are considered speaker characteristics and will be used to analyse the performance of the model for the different groups.

It is also possible to add utterance-level characteristics, such as emotion. They must be preceded by `utt_`; they will not be propagated through the speakers.

One last requirement; the utterances of each speaker must appear each after the other in the datafile. This is required by the function `split_trials_enrolls`, used in the ASV evaluation to split the data into trial and enrolment utterances. This function iterates over the datafile assuming that once the speaker changes, it will not see the previous speaker again.
