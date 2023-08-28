import json
import os
import logging


LOGGER = logging.getLogger("progress")


def add_root(
    manifest_filepath: str,
    root_folder: str,
    log_dir: str,
    min_dur: float = None,
    max_dur: float = None,
):
    """
    Add the root folder to the paths of the audiofiles and store the resulting
    manifest files within the experiment folder.

    Args:
        manifest_filepath (str): path to the manifest file
        root_folder (str): path to the root folder
        log_dir (str): path to the experiment folder
        min_dur (float, optional): minimum duration of the audiofiles to keep.
        max_dur (float, optional): maximum duration of the audiofiles to keep.
    """
    LOGGER.info(f"Adding root `{root_folder}` to datafiles `{manifest_filepath}`")
    new_paths = list()
    if isinstance(manifest_filepath, str):
        manifest_filepath = [manifest_filepath]
    for fpath in manifest_filepath:
        LOGGER.info(f"Adding root `{root_folder}` to `{fpath}`")
        new_paths.append(os.path.join(log_dir, fpath))
        if os.path.exists(new_paths[-1]):  # manifest already modified
            LOGGER.info(f"New filepath `{new_paths[-1]}` already exists; skipping")
            continue
        os.makedirs(os.path.dirname(new_paths[-1]), exist_ok=True)
        too_short, too_long, included = list(), list(), list()
        with open(new_paths[-1], "w") as writer:
            with open(fpath) as reader:
                for line in reader:
                    obj = json.loads(line)
                    dur = obj["duration"]
                    if min_dur is not None and dur < min_dur:
                        too_short.append(dur)
                        continue
                    if max_dur is not None and dur > max_dur:
                        too_long.append(dur)
                        continue
                    included.append(dur)
                    obj["audio_filepath"] = obj["audio_filepath"].replace(
                        "{root}", root_folder
                    )
                    writer.write(json.dumps(obj) + "\n")

        LOGGER.info(
            f"{len(included)} samples included ({round(sum(included) / 3600, 3 )} h)"
        )
        LOGGER.info(
            f"{len(too_short)} samples too short ({round(sum(too_short) / 3600, 3 )} h)"
        )
        LOGGER.info(
            f"{len(too_long)} samples too long ({round(sum(too_long) / 3600, 3 )} h)"
        )

    LOGGER.info(f"New datafiles: `{new_paths}`")
    return new_paths
