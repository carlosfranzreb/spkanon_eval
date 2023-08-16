# StarGANv2-VC

StarGANv2-VC is an unsupervised voice conversion (VC) model where the input speech is adapted to conform with a given style vector, which we extract from the mapping network with the target ID. The generator receives the style vector of the target, as well as the spectrogram and the F0 contour of the input speech. It converts the input spectrogram to the given style vector, preserving the F0 contour, which is then synthesized with the Parallel WaveGAN. The generator is trained adversarially with two models: one which discriminates real samples from fake ones, and another which recognizes the converted sample's speaker. Six additional losses are included to ensure that the style vector is used appropriately, and that the converted speech is consistent with the input speech.

- Interspeech 2021 paper: https://www.isca-speech.org/archive/interspeech_2021/li21e_interspeech.html
- GitHub repository: https://github.com/yl4579/StarGANv2-VC

## Installation

To run this pipeline, you have to add the original StarGANv2-VC repository as a submodule and download the weights. You can do so by running the build file `build/stargan.sh`.

## Implementation

We have implemented wrappers for its different components and included each in the appropriate phase. You can find the whole configuration for this model in `config/pipelines/stargan.yaml`.

### Feature extraction

The feature extraction module comprises the spectrogram extractor. This component is described [here](components/featex/spectrogram.md).

### Feature processing

The feature processing module comprises the StarGANv2-VC conversion component. It receives spectrograms and the source speaker labels as input, and optionally the desired target speaker. It outputs the converted spectrogram, conditioned on the target speaker. If a target speaker is not given, it uses its own target selection algorithm to pick a target. You can read more about target selection algorithms [here](components/target_selection.md). We use the mapping network to extract the style vectors of the target speakers.

### Synthesis

The Parallel WaveGAN synthesizer comes from a [Python package](https://pypi.org/project/parallel-wavegan/). We implement a wrapper for it, which receives the converted spectrogram as input and outputs the synthesized waveform.

## Training data

The model was trained on 20 speakers of VCTK, a corpus of read speech recorded in a professional studio. The exact dataset can be found in the GitHub repository, linked above.

## Results

We have evaluated these models with our [default setup](components/evaluation/default.md). The experiment folder for this run is `wandb/run-20230718_132607-msdyh7v8`. The following tables result from running the script `scripts/print_results`.

### Evaluation results

| | cv-test_3utts | ls-test-clean | edacc-test | ravdess |
| --- | --- | --- | --- | --- |
| whisper-small | 0.71 | 0.14 | 0.75 | 0.57 |
| whisper-large | 0.51 | 0.09 | 0.6 | 0.35 |
| ser-audeering-w2v | 0.99 | 1.0 | 0.99 | 0.98 |
| nisqa | 3.41 | 3.12 | 2.95 | 3.32 |
| asv-plda/ignorant/results | 0.27 | 0.32 | 0.38 | 0.44 |
| asv-plda/lazy-informed/results | 0.22 | 0.27 | 0.28 | 0.36 |

### Inference time for 10s input duration (in seconds)

| Component | inference time |
| --- | --- |
| cpu_inference | 8.71 |
| cuda_inference | 0.16 |
