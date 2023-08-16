# Speaker Anonymization

![Workflow badge](https://github.com/carlosfranzreb/spkanon/actions/workflows/build.yml/badge.svg)

Evaluation framework for speaker anonymization systems. [Here](docs/index.md) we describe how the framework works. Please write an issue if anything is unclear or you need help, and leave a star if you find this useful! Contributions are welcome, including adding new components, improving the framework or the documentation.

## Installation

To install the necessary packages, run the `build/framework.sh` script. It will create a Conda environment inside the repository, where the necessary dependencies are installed. Once the script is done, you can run the tests with `python -m unittest discover -s ./tests`.

Then you need to define the datasets to use and the root directory where the datasets are stored in the config. If you do so in the `config/datasets/evaluation.yaml` file, you can directly evaluate an STT-TTS that uses Whisper and FastPitch by running `python run.py --config config/full.yaml`. You can read more about that pipeline [here](docs/pipelines/whisper_fastpitch.md).

## Anonymization pipelines

- STT-TTS with Whisper & FastPitch: does not require any further installation. Read more about this pipeline [here](docs/pipelines/whisper_fastpitch.md).
- StarGANv2-VC: install it with `build/stargan.sh`. Read more about this pipeline [here](docs/pipelines/whisper_fastpitch.md).
- SoftVC: : install it with `build/softvc.sh` does not require any further installation. Read more about this pipeline [here](docs/pipelines/whisper_fastpitch.md).

## Results

Full results for the three pipelines mentioned above can be found under the `logs` folder. They are gathered in the notebooks under `scripts/spsc2023_results`.

## Citation

```tex
@inproceedings{franzreb2023comprehensive,
  title={A Comprehensive Evaluation Framework for Speaker Anonymization Systems},
  author={Franzreb, Carlos and Polzehl, Tim and Moeller, Sebastian},
  booktitle={Proc. 3rd Symposium on Security and Privacy in Speech Communication},
  year={2023},
}
```
