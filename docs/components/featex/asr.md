# Automatic Speech Recognition (ASR)

ASR is used to assess the performance of text-to-speech and voice conversion models in many publications, and also in the Voice Privacy Challenge. It determines whether the anonymization system preserves the linguistic information.

We have implemented Whisper, currently one of the best ASR models. It is an encoder-decoder Transformer trained on 680,000 hours of multilingual speech scraped from the Internet. The size and inherent diversity of the dataset make the resulting model robust to noise, accent and choice of words.

It can be used both as part of an anonymization pipeline in the feature extraction module, to extract either the encoding or the text from a given speech sample, or as an evaluation component, to compute the WER of the anonymized speech. An example of its use as an anonymization component can be found in the config `config/whisper_fastpitch.yaml`; to use it as an evaluation component, check the configs under `config/components/asr`, which you can add to your experiment config under the `eval.components` key (see [this documentation](components.md) for more details on how to do so).
