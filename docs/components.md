# Components

Components are the building blocks of the framework. Each module is assigned one or several components through the configuration file. Components may require different configuration parameters and ingest different data; all of this is defined in the configuration file.

## List of components

Here is a list of the existing components, with links to their documentations. Components that are specific to the **evaluation pipeline** can be found [here](evaluation.md).

### Feature Extraction

- [SpkId](components/featex/spkid.md): speaker recognition
- [ASR](components/featex/asr.md): automatic speech recognition
- [HuBERT (Soft-VC)](pipelines/softvc.md): feature extraction component of the Soft-VC anonymization pipeline
- [Spectrogram](components/featex/spectrogram.md): computes the spectrogram of the waveforms

### Feature Processing

- [StarGANv2-VC](pipelines/stargan.md): performs voice conversion given an input spectrogram and (optionally) a target speaker
- [FastPitch](pipelines/whisper_fastpitch.md): transforms text into a spectrogram, conditioned on a target speaker
- [Soft-VC's acoustic model]((pipelines/softvc.md)): transforms HuBERT features into a spectrogram

#### Target Selection

The feature processing module may require the selection of a target speaker. This is defined in the config, and the feature processing component initializes it and stores it. The target selection algorithm may be enforced to pick consistent targets, meaning that each source speaker is always given the same target. You can find more about the implemented algorithms and how it works [here](components/target_selection.md).

### Feature Fusion

- [Feature concatenation](components/featfusion/concat.md): concatenates the given features

### Speech Synthesis

- [NeMo HifiGAN](pipelines/whisper_fastpitch.md): trained by NeMo on 10 HiFiTTS speakers
- [Parallel WaveGAN](pipelines/stargan.md): wrapper for the synthesizers of the parallel wavegan repository
- [Soft-VC's HiFiGAN](pipelines/softvc.md): synthesizer fine-tuned on 1 LibriTTS speaker

## Implementing a new component

When designing a new component, it must include two methods:

1. `__init__(self, config)`
2. `run(self, batch)`

It may include the following methods, which will be called from the outside if defined in the configuration file:

1. `train(self, exp_folder, *args)`
2. `finetune(self, exp_folder, *args)`
3. `eval_dir(self, exp_folder)`

### Required methods

They are all implemented following the same architecture.

#### Initialization (`__init__(self, config, device, **kwargs))`)

When the anonymization pipeline starts, all components are initialized with the `setup_module.py` script. It requires the configuration file to contain a `cls` parameter, to know which class to import. The component's config must also include all the parameters necessary for its initialization. Each component has an example configuration in `config/components`. Alternatively, you can find which parameters are required by a component in its `__init__` method.

Once imported, it initializes the class with the config and optionally with the [PTL Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html). The PTL Trainer is passed to the component's `__init__` function if it is passed to the `setup` function.

#### Run (`run(self, batch)`)

This method is the only required one other than the initialization method. It receives a batch as input, whose structure varies depending on the module. The task of this method is to run the batch (or the relevant part of it) through the component and return the output.

### Optional methods

As mentioned above, there are only two methods that are required: `__init__` and `run`. There are three further methods that may be called from the outside, if defined in the config: `train`, `finetune` and `eval_dir`.

#### Training (`train(self, exp_folder, *args)`)

This method trains the component's model (or models) from scratch. It will store the resulting models in the given experiment folder, as well as update its attributes.

It may require additional arguments whose values are generated on run time and can therefore be not included in the config file.

#### Fine-tuning (`finetune(self, exp_folder, *args)`)

This method is similar to `train`, but the training is not performed from scratch.

#### Evaluation (`eval_dir(self, exp_folder)`)

This method runs all the data found in the given experiment folder through the component and dumps an evaluation report. If the component is an ASR model, it will compute the WER of each audio file in the experiment folder and dump each path along with its WER in an audio file.

Which files are evaluated (original, anonymized, or both) is defined in the config file if necessary. Some evaluations are a direct comparison between the original and the anonymized data.
