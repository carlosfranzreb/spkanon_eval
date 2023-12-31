import os
import string
import logging

import whisper
from whisper.normalizers import EnglishTextNormalizer
import torch
import editdistance
import numpy as np

from spkanon_eval.datamodules.dataloader import eval_dataloader
from spkanon_eval.evaluation.analysis import analyse_results
from spkanon_eval.featex.asr.whisper_analysis_utils import analyse_func, headers_func
from spkanon_eval.component_definitions import InferComponent, EvalComponent


SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
LOGGER = logging.getLogger("progress")


class Whisper(InferComponent, EvalComponent):
    def __init__(self, config, device, **kwargs):
        self.model = whisper.load_model(
            config.size,
            download_root="checkpoints/whisper",
        ).to(device)
        self.model_size = config.size
        self.device = device
        self.out = config.output
        self.config = config

        # update batch size; data is only needed by `eval_dir`
        if "data" in self.config:
            self.config.data.config.batch_size = config.batch_size

    def run(self, batch):
        """
        1. pad each audio to span 30 seconds: whisper expects log-mel spectrograms
            that span 30 seconds as input
        2. compute log-mel spectrograms
        3. return the predicted text or the encoder output
        """
        mels = list()
        for i in range(batch[0].shape[0]):  # iterate over waveforms in batch
            padded = whisper.pad_or_trim(batch[0][i])
            mels.append(whisper.log_mel_spectrogram(padded).unsqueeze(0))
        mels = torch.cat(mels, dim=0)
        if self.out == "text":  # return predicted text
            out = self.model.decode(
                mels, options=whisper.DecodingOptions(fp16=False, language="en")
            )
            return [decoding.text for decoding in out]
        elif self.out == "encoding":  # return encoder output
            return self.model.encoder(mels)

    def train(self, exp_folder: str) -> None:
        raise NotImplementedError("Whisper wrapper does not support training")

    def eval_dir(self, exp_folder: str, datafile: str, *args) -> None:
        """
        Compute the transcription of each sample and its WER. Dump both into a file
        within the current experiment folder. Keep track of the average WER of each
        datafile and each speaker characteristic (age, gender, etc). Dump these
        averages as well.

        Args:
            exp_folder: path to the experiment folder
            datafile: datafile to evaluate
        """
        LOGGER.info("Computing WER of eval data with dataloader")
        if self.out != "text":
            self.out = "text"
        normalizer = EnglishTextNormalizer()
        dump_folder = os.path.join(exp_folder, "eval", f"whisper-{self.model_size}")
        os.makedirs(dump_folder, exist_ok=True)
        stats = {"n_edits": list(), "n_words_ref": list()}

        # define the dump file and write the headers
        dump_file = os.path.join(dump_folder, os.path.basename(datafile))
        with open(dump_file, "w", encoding="utf-8") as f:
            f.write("path n_edits n_words_ref wer text\n")

        for _, batch, sample_data in eval_dataloader(
            self.config.data.config, datafile, self.device
        ):
            texts_pred = self.run(batch)  # compute the transcriptions for the batch
            for i, text_pred in enumerate(texts_pred):  # iterate through the batch
                # compute the WER for the current sample
                audiofile = sample_data[i]["path"]
                text_ref = sample_data[i]["text"]
                n_edits, n_words, wer = compute_edits(normalizer(text_pred), text_ref)
                # if wer could not be computed, skip
                if n_words == 0:
                    LOGGER.warn(f"Reference text of {audiofile} has no words; WER = 0")
                # dump the results for this sample into the datafile
                with open(dump_file, "a", encoding="utf-8") as f:
                    f.write(f"{audiofile} {n_edits} {n_words} {wer} {text_pred}\n")
                # update datafile stats
                stats["n_edits"].append(n_edits)
                stats["n_words_ref"].append(n_words)

        analyse_results(
            dump_folder,
            datafile,
            [np.array(stats["n_edits"]), np.array(stats["n_words_ref"])],
            analyse_func,
            headers_func,
        )

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.model.to(device)
        self.device = device


def compute_edits(text_pred, text_ref):
    """
    Normalize the texts, split them into words and compute the Levenhstein
    distance between the lists of words. Return the edit distance, the number of words
    of `text_ref` (which we assume is the reference) and the WER.
    If the reference text has no words, return all zeros so that the analysis may
    be run.
    """
    text_pred = normalize_text(text_pred)
    text_ref = normalize_text(text_ref)
    n_edits = editdistance.eval(text_pred, text_ref)
    if len(text_ref) > 0:
        return n_edits, len(text_ref), round(n_edits / len(text_ref), 2)
    else:
        return 0, 0, 0


def normalize_text(text):
    """
    Normalize the text by removing the punctuation and making it lowercase.
    Split the text into words and return the list of words.
    """
    return text.translate(str.maketrans("", "", string.punctuation)).lower().split()


def dump_averages(out_dir, datafile, subset, data):
    """
    Add the data to the appropriate file. The file is determined by the subset and is
    created under `out_dir` if it does not exist. The data is added to the file as a
    new line.
    """
    # define the columns used for this subset; `totals` has no subset column
    cols = ["datafile", "n_edits", "n_words_ref", "wer"]
    if subset != "totals":
        cols.insert(1, subset)

    # define the filename and create it with headers if needed
    fname = os.path.join(out_dir, f"avg_wers-{subset}.txt")
    if not os.path.exists(fname):
        with open(fname, "w") as f:
            f.write(" ".join(cols) + "\n")

    # define a subroutine that writes data to the file
    def write_data(data):
        data["datafile"] = datafile
        data["wer"] = round(data["n_edits"] / data["n_words_ref"], 2)
        with open(fname, "a") as f:
            f.write(" ".join([str(data[col]) for col in cols]) + "\n")

    # write the data for this subset; for speaker chars., iterate over the values
    if subset == "totals":
        write_data(data)
    else:
        for subset_val, val_data in data.items():
            val_data[subset] = subset_val
            write_data(val_data)
