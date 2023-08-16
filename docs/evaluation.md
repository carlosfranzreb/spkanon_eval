# Evaluation

The evaluation module constitutes its own pipeline, and is run separately from the anonymization pipeline. Its configuration is defined under the key `eval`. An evaluation pipeline requires anonymized data, meaning that an anonymization pipeline must be run beforehand. However, once we have run an anonymization pipeline, we can run evaluation pipelines independently by defining in the configuration file which experiment folder should be evaluated. This is done with the `exp_folder` config parameter. If this parameter is set to `null`, the evaluation pipeline will use the anonymized data of the current anonymization pipeline. It must therefore be run in the same job as the anonymization pipeline.

## Evaluation components

- [ASV-VPC](components/evaluation/spkid_plda.md): Automatic Speaker Verification system based on speaker embeddings and a PLDA algorithm.
- [ASR](components/featex/asr.md): Automatich Speech Recognition.
