# Speaker Recognition

Speaker recognition models, also known as speaker verification or speaker identification models, are designed to identify and distinguish between different individuals based on their unique vocal characteristics. Here, they can be used both as a feature extraction component and for evaluation purposes (i.e. [ASV evaluation](components/evaluation/spkid_plda.md)).

Currently, we have implemented wrappers for the speaker recognition models of SpeechBrain. Configurations for all these models can be found under `config/components/spkid`. SpeechBrain has trained an [x-vector model](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) and an [ECAPA-TDNN model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) on VoxCeleb 1&2.