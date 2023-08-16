# Spectrogram Extraction

The `SpecExtractor`, defined in the file `src/featex/spectrogram.py`, is class is designed to compute the mel-spectrogram of an audio tensor after normalizing it. Upon initialization, the `SpecExtractor` object sets up the mel-spectrogram transformation using the `torchaudio.transforms.MelSpectrogram` class. The mel-spectrogram is a representation of the audio signal's spectral content over time and is widely used in audio analysis tasks.

The `config` object should contain the following attributes:

- `n_mels` (int): The number of mel filterbanks to be used in the computation.
- `n_fft` (int): The number of data points in each Fourier Transform.
- `win_length` (int): The window size for the short-time Fourier Transform (STFT).
- `hop_length` (int): The hop size (stride) for the STFT.

You can find an example of how it is used in the [StarGANv2-VC pipeline](pipelines/stargan.md).
