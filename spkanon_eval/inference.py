import os
import json
import logging

from torch.cuda import OutOfMemoryError
import torchaudio

from spkanon_eval.datamodules.dataloader import eval_dataloader


LOGGER = logging.getLogger("progress")


def infer(model, exp_folder, datafiles, dump_dir, infer_input, data_cfg, sample_rate):
    """
    - Run each recording through the model and store the resulting audiofiles in
        `dump_dir`.
    - We iterate through the audiofiles with the eval_dataloader, which returns
        paths and batches (the paths are used to store the results).
    - Once the batch is anonymized, each sample is trimmed and saved to the
        corresponding path.
    - After the inference is finished, we create datafiles for the anonymized data,
        which can be used to initialize a dataloader. They are also dumped to
        `dump_dir`.
    """

    LOGGER.info(f"Running inference with files {datafiles}")
    LOGGER.info(f"Dumping anonymized audiofiles to `{dump_dir}`")

    # iterate through the datafiles, anonymize them and dump the results
    for _, batch, data in eval_dataloader(data_cfg, datafiles, model.device):
        # anonymize the batch
        try:
            audio_anon, target = model.infer(batch, data, infer_input)
        except OutOfMemoryError:
            LOGGER.error(f"OOM Error on batch with data {data}")
            continue

        for i in range(len(audio_anon)):
            # compute the path to the anonymized audiofile
            audio_path = data[i]["audio_filepath"][len(data_cfg.root_folder) + 1 :]
            format = os.path.splitext(audio_path)[1][1:]
            file_anon = os.path.join(dump_dir, audio_path)
            # save the anonymized audio to the computed path
            os.makedirs(os.path.split(file_anon)[0], exist_ok=True)
            torchaudio.save(file_anon, audio_anon[i], sample_rate, format=format)

    LOGGER.info("Done. Creating datafiles for the anonymized data")
    # create datafiles for the anonymized data by copying and modifying the originals
    for read_file in datafiles:
        # open the original datafile
        with open(read_file, "r") as reader:
            # create a new datafile for the anonymized data
            write_file = read_file.replace(exp_folder, dump_dir)
            LOGGER.info(f"Creating datafile `{write_file}`")
            os.makedirs(os.path.split(write_file)[0], exist_ok=True)
            with open(write_file, "w") as writer:
                # iterate through the lines of the original datafile
                for line in reader:
                    # load the data and modify the audiofile path
                    data = json.loads(line)
                    data["audio_filepath"] = data["audio_filepath"].replace(
                        data_cfg.root_folder,
                        os.path.join(dump_dir),
                    )
                    # check that the anonymized filepath exists
                    if not os.path.exists(data["audio_filepath"]):
                        fpath = os.path.basename(data["audio_filepath"])
                        LOGGER.warn(f"File '{fpath}' was not anonymized")
                        continue
                    # dump the modified data into the new datafile
                    writer.write(json.dumps(data) + "\n")
