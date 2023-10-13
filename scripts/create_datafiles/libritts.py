"""
Create the data file as is expected by the dataset: as a txt file with one
dict per line, with the fields "path": str, "duration": float and "label":
int. Given our use case, we store the speaker ID in the label field. We also store
the transcript, which may be used in the evaluation.
"""


import os
import json
from argparse import ArgumentParser
import torchaudio


def create_file(folder, dump_file, root_folder):
    """
    Create a datafile for the Librispeech dataset stored in "folder" and dump
    the file in "dump_file". The LibriTTS folder contains one folder per speaker,
    and each of those folders, contains a chapter, and that chapter folder contains
    WAV files.

    The root folder is replaced with the {root} marker
    """
    writer = open(dump_file, "w")
    for dirpath, _, filenames in os.walk(folder):
        for fname in filenames:
            if fname.endswith(".wav"):
                audiofile = os.path.join(dirpath, fname)
                audio, sample_rate = torchaudio.load(audiofile)
                text_file = f"{os.path.splitext(audiofile)[0]}.original.txt"
                writer.write(
                    json.dumps(
                        {
                            "path": audiofile.replace(root_folder, "{root}"),
                            "text": open(text_file).read(),
                            "duration": audio.shape[1] / sample_rate,
                            "label": fname.split("_")[0],
                        }
                    )
                    + "\n"
                )
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--dump_file")
    parser.add_argument("--root_folder")
    args = parser.parse_args()
    create_file(args.folder, args.dump_file, args.root_folder)
