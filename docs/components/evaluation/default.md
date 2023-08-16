# Default evaluation setup

The default evaluation setup is used to compare anonymization systems fairly, aiming to cover as many use cases as possible.

## Datasets

The datasets are split in two: evaluation data, which is anonymized and afterwards evaluated, and training data, which is used to train the evaluation components. Currently, the ASV system is the only one that uses the training data. The datasets are defined in the config file `config/datasets/evaluation.yaml`.

Samples of durations below 2 seconds or above 30 seconds are filtered out on runtime, and the amount of removed data is logged to the progress file.

### Evaluation data

- LibriSpeech test-clean: `data/librispeech/ls-test-clean.txt`
- EdAcc test: `data/edacc/edacc-test.txt`
- Common Voice test: `data/common_voice/cv-test_3utts.txt` (speakers that comprise less than 3 utterances are removed)
- RAVDESS: `data/ravdess/ravdess.txt`

### Training data

The following dataset are used to train the evaluation components (if needed).

- LibriSpeech train-clean-100: `data/librispeech/ls-train-clean-100.txt`
- EdAcc dev: `data/edacc/edacc-dev.txt`

## Evaluation components

### Privacy

- Ignorant ASV system based on x-vectors: `config/components/asv/vpc_ignorant.yaml`
- Lazy-informed ASV system based on x-vectors: `config/components/asv/vpc_lazy-informed.yaml`

### Utility

- Large ASR: `config/components/asr/whisper_large.yaml`
- Small ASR: `config/components/asr/whisper_small.yaml`
- SER: `config/components/ser/audeering_w2v.yaml`
- Naturalness: `config/components/naturalness/nisqa.yaml`
- Performance: `config/components/performance/performance.yaml`