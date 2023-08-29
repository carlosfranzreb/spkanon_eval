# Speech Emotion Recognition (SER)

We use a SER model to measure the emotion preservation of the anonymization model, by computing the cosine similarity between the emotion embeddings of the original and the anonymized speech. The embeddings are computed with an SER model based on Wav2vec2.0 and fine-tuned on the MSP-Podcast dataset, which has annotated the valence, arousal and dominance dimensions of each sample. You can read more about the SER model in the [HuggingFace page](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim), and also in the paper cited below.

## Citation

```tex
@article{wagner2023dawn,
    title={Dawn of the Transformer Era in Speech Emotion Recognition: Closing the Valence Gap},
    author={Wagner, Johannes and Triantafyllopoulos, Andreas and Wierstorf, Hagen and Schmitt, Maximilian and Burkhardt, Felix and Eyben, Florian and Schuller, Bj{\"o}rn W},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    pages={1--13},
    year={2023},
}
``` 
