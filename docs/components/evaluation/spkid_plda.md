# Automatic Speaker Verification (ASV)

The task of speaker verification is to determine whether two utterances belong to the same speaker. The dataset is usually split into trial and enrollment utterances, where each speaker usually has at least one trial utterances and several enrollment utterances. All trial utterances are compared with all enrollment utterances with the ASV system, which outputs a likelihood of the two utterances belonging to the same speaker.

Where to draw the decision boundary between matches and non-matches is determined with a threshold. To evaluate the performance of ASV systems, the Equal Error Rate (EER) is used. The EER is the threshold for which both false positives and false negatives are equal.

Our approach is based on the Voice Privacy Challenge, which uses a [PLDA model](https://towardsdatascience.com/probabilistic-linear-discriminant-analysis-plda-explained-253b5effb96) on top of speaker embeddings. PLDA is a probabilistic extension of [LDA](https://scikit-learn.org/stable/modules/lda_qda.html), which is a dimensionality reduction algorithm that computes a projection of the data where the between-class covariance is maximized and the within-class covariance minimized. PLDA extends this paradigm to unseen classes, which is the case in ASV. Given a data point that was unseen during training, PLDA can compute the distribution it belongs to, i.e. its class. It can do so even if the class was not previously present in the data.

In ASV, each speaker represents a class. The PLDA model is trained with a separate dataset, whose speakers are not present in the evaluation dataset. Once trained, the PLDA model is used to compute the [log-likelihood ratio (LLR)](https://towardsdatascience.com/the-likelihood-ratio-test-463455b34de9) of two utterances belonging to the same speaker. The LLR is a floating number. If it is positive, the model believes that the utterances belong to the same speaker.

Our ASV implementation can be run with any speaker embedding models from SpeechBrain or NeMo. You can find their configurations under `config/components/spkid`. Configurations that include each of these speaker embeddings, as well as the different attack scenarios explained below, can be found under `config/components/asv`.

## Training phase

This phase requires a dataset different from the one used for evaluation.

1. [optional] Remove utterances that are shorter than 4 seconds, and remove speakers that have less than 2 utterances.
2. Fine-tune the [SpkId net](../featex/spkid.md).
3. Extract SpkId vectors for each utterance from the fine-tuned net.
4. Center the SpkId vectors.
5. [optional] Decrease the dimensionality of the centered SpkId vectors with [sklearn's LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).
6. Train Ravi B. Sojitra's [PLDA model](https://github.com/RaviSoji/plda) with the vectors resulting from LDA.

## Evaluation phase

1. Split the evaluation data into trial and enrollment data. The last utterance of each speaker is the trial utterance.
2. Use anonymized utterances of trial data, and maybe anonymized enrollment data (as configured with `anon_data`). This option implements the two attack options defined by the VPC.
3. Compute the SpkId vectors of all trial and enrollment utterances with the model fine-tuned in the training phase.
4. Center the SpkId vectors and decrease their dimensionality with the trained LDA model
5. Compute LLRs of all pairs of trial and enrollment utterances with the trained PLDA model. For this we use the method to compute marginal likelihoods from the [PLDA model](https://github.com/RaviSoji/plda).
6. Compute the [ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) with sklearn and the EER. The plot, the threshold and the EER are dumped to the experiment folder.

## Attack scenarios

We consider two attack scenarios:

1. Ignorant: the attacker does not have access to the anonymization model. The training data of the ASV system and the enrollment data are not anonymized.
2. Lazy-informed: the attacker has access to the anonymization model. The training data of the ASV system is anonymized without consistent targets, and the enrollment data is anonymized with consistent targets. Target speakers are consistent when each source speaker is always assigned the same target.

## Configuration

If you want to evaluate your anonymization pipeline with this component, you have to include it in your configuration file under `eval.components`. Its configuration looks as follows.

```yaml
asv:
  cls: spkanon_eval.evaluation.asv_vpc.ASV
  scenario: ignorant # attack scenario; can be ignorant or lazy-informed
  train: true
  data: ${data}
  anon_data: false # whether training data should be anonymized before training the model
  baseline: ${eval.config.baseline}
  # reduced_dims: 2 # (optional) no. of dimensions to reduce the spkid vectors to (VPC uses 200)
  lda_ckpt: null
  plda_ckpt: null
  filter: # (optional) filter out samples that don't meet these criteria
    min_dur: 4 # min. duration (in s) for each utterance
    min_samples: 2 # min. no. of samples per speaker
  spkid:
    cls: spkanon_eval.featex.spkid.SpkId # cls that extracts this feature
    path: ecapa_tdnn # path to ckpt or name of pretrained model
    train: false
    trainer_cfg: config/trainers/spkid.yaml # cfg used to init the trainer and optimizer
    resources: ${resources}
```

Both the filtering of training data and the LDA dimensionality reduction are optional. The *lazy-informed* attack scenario involves anonymizing training and enrollment data, and it therefore needs to further parameters:

```yaml
  inference: # keys required by the anonymization model to find the data it requires
    input:
      spectrogram: spectrogram
      target: target
  sample_rate: ${synthesis.sample_rate} # output sample rate of the anonymization model
```

Training the ASV system is optional. To disable it, change the value of `train` to `false`. When training is disabled, checkpoints of the models required by the ASV system must be defined (`plda_ckpt` and maybe `lda_ckpt`). You can find example configurations for the two attack scenarios, both with enabled and disabled training in the directory `config/components/asv`.
