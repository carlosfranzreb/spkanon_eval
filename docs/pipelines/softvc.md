# Soft-VC

Soft-VC is a voice conversion model based on HuBERT. It computes the HuBERT features of the input speech, which implicitly perform the anonymization, given that the HuBERT model was trained to predict text, and not the speaker. They improve the preservation of the linguistic content by predicting the distribution over the discrete HuBERT units, resulting in what they term soft speech units. These units capture more paralinguistics and improve the intelligibility and naturalness of the converted speech. In the Soft-VC pipeline, HuBERT is followed by an acoustic model that transforms HuBERT features into a spectrogram, which is then synthesized by HiFI-GAN. The HiFi-GAN they used is only trained on one speaker from LibriTTS, making Soft-VC an any-to-one voice conversion model.

- ICASSP 2022 paper: https://ieeexplore.ieee.org/abstract/document/9746484
- GitHub repository: https://github.com/bshall/soft-vc

## Installation

To run this pipeline, you have to add the original repositories as submodules. You can do so by running the build file `build/softvc.sh`.

## Implementation

We have implemented wrappers for its different components and included each in the appropriate phase. You can find the whole configuration for this model in `config/pipelines/softvc.yaml`.

### Feature extraction

The feature extraction module comprises the HuBERT model, followed by a linear projection to predict probabilities over the different speech units. This component is implemented in a separate GitHub repository (<https://github.com/bshall/hubert>). Our wrapper can be found in the file `src/featex/hubert_softvc.py`.

### Feature processing

This module comprises the acoustic model that transforms the HuBERT units to a spectrogram. It is implemented in a separate GitHub repository (<https://github.com/bshall/acoustic-model>), and trained on one speaker of LJSpeech. Therefore, no target selection is required for this pipeline. Our wrapper can be found in the file `src/featproc/acoustic_softvc.py`.

### Synthesis

The feature processing module comprises the acoustic model that transforms the HuBERT units to a spectrogram. It is implemented in a separate GitHub repository (<https://github.com/bshall/hifigan>), and trained with the output of the acoustic model. Our wrapper can be found in the file `src/synthesis/hifigan_softvc.py`.

## Training data

HuBERT is a self-supervised model trained on the whole LibriSpeech dataset (960 h). The acoustic model and the vocoder are trained with one speaker of LJSpeech (xx h). The exact train and test splits can be found [here](https://github.com/bshall/hifigan/releases/tag/v0.1). The acoustic model is trained for 50k steps. The checkpoint with the lowest validation loss is chosen. The HiFiGAN is trained with ground-truth spectrograms for 1M steps and then fine-tune on predicted spectrograms for 500k steps.

## Results

We have evaluated these models with our [default setup](components/evaluation/default.md). The experiment folder for this run is `wandb/run-20230718_091956-h0lspxgz`. The following tables result from running the script `scripts/print_results`.

### Evaluation results

| Component | cv-test_3utts | edacc-test | ls-test-clean | ravdess |
| --- | --- | --- | --- | --- |
| asv-plda/ignorant/results | 0.43 | 0.45 | 0.42 | 0.39 |
| asv-plda/lazy-informed/results | 0.27 | 0.30 | 0.19 | 0.10 |
| nisqa | 3.48 | 3.60 | 3.95 | 3.92 |
| ser-audeering-w2v | 1.00 | 1.00 | 1.00 | 1.00 |
| whisper-large | 0.29 | 0.45 | 0.06 | 0.08 |
| whisper-small | 0.36 | 0.53 | 0.07 | 0.16 |

### Inference time for 10s input duration (in seconds)

| Component | inference time |
| --- | --- |
| cpu_inference | 4.91 |
| cuda_inference | 0.43 |
