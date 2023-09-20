import logging

import torch

from spkanon_eval.setup_module import setup


LOGGER = logging.getLogger("progress")


class Anonymizer:
    def __init__(self, config, log_dir):
        super().__init__()
        self.config = config
        self.device = config.device
        self.log_dir = log_dir

        for module in ["featex", "featproc", "featfusion", "synthesis"]:
            cfg = config.get(module, None)
            if cfg is not None:
                setattr(self, module, setup(cfg, self.device))
            else:
                setattr(self, module, None)

        # if possible, get the output of the featproc module
        if self.featproc is not None:
            self.proc_out = self.featproc.pop("output")

        # if there is a target selection algorithm, pass it to the right component
        target_selection_cfg = config.get("target_selection", None)
        if target_selection_cfg is not None and self.featproc is not None:
            args = [target_selection_cfg]
            if hasattr(target_selection_cfg, "extra_args"):
                for arg in target_selection_cfg.extra_args:
                    args.append(getattr(self, arg))
            for name, component in self.featproc.items():
                if hasattr(component, "target_selection"):
                    LOGGER.info(f"Passing target selection algorithm to {name}")
                    component.init_target_selection(*args)

    def get_feats(self, batch, source, target=None):
        """
        Run the featex, featproc and featfusion modules. Returns spectrograms and
        targets. Sources refer to the input speaker, and targets to the output speaker.
        They are both added to the batch after the feature extraction phase with the
        keys `source` and `target`. If the `target` argument is not None, it is created
        with -1 values, which indicate that no target has been defined for the
        corresponding sample.
        """
        out = batch
        if self.featex is not None:
            out = self._run_module(self.featex, batch)

        # add targets and sources to the batch
        if target is None:
            target = torch.ones(batch[0].shape[0], dtype=torch.int64) * -1
            target = target.to(batch[0].device)
        out["target"] = target
        out["source"] = source

        if self.featproc is not None:
            processed = self._run_module(self.featproc, out)
            out_proc = dict()
            for feat in self.proc_out["featex"]:
                out_proc[feat] = out[feat]
            for feat in self.proc_out["featproc"]:
                out_proc[feat] = processed[feat]
            out = out_proc
        if self.featfusion is not None:
            out = self.featfusion.run(out)
        return out

    def forward(self, batch, source, target=None):
        """Returns anonymized speech as a tensor (B, S, T)."""
        fused = self.get_feats(batch, source, target)
        return self.synthesis.run(fused)

    def infer(self, batch, data, inputs):
        """
        Run the forward pass to get the synthesized speech and unpad the audios before
        returning them. We do so by looking for repeated values at the end of the
        spectrogram, which indicates the padding. The audios are detached from the
        graph and moved to CPU.

        ! This method requires the output of `get_feats` to be a spectrogram.
        """
        source = [d["label"] for d in data]
        feats = self.get_feats(batch, source)
        audios = self.synthesis.run(feats).detach().cpu()
        specs = feats[inputs.spectrogram]
        unpadded_audios = list()
        for i in range(specs.shape[0]):
            try:
                # find the number of repeated values at the end of each channel
                min_idx = list()
                for channel in specs[i]:
                    same_value = channel == channel[-1]
                    for idx in range(same_value.shape[0] - 1, 0, -1):
                        if same_value[idx] == False:
                            break
                        last_idx = idx
                    min_idx.append(last_idx)
                # compute the ratio between the spectrogram length and the audio length
                ratio = audios.shape[-1] / specs[i].shape[-1]
                # remove max(min_idx) frames from the spec
                removed_frames_spec = specs[i].shape[1] - max(min_idx)
                # compute the number of samples to remove from the audio
                removed_frames_audio = removed_frames_spec * ratio
                # remove the padding from the audio
                unpadded_audios.append(audios[i, :, : -int(removed_frames_audio)])
            except Exception:
                fpath = data[i]["audio_filepath"]
                LOGGER.error(
                    f"Error while unpadding anonymized `{fpath}`; dumping padded"
                )
                unpadded_audios.append(audios[i])
        return unpadded_audios, feats[inputs.target]

    def _run_module(self, module, batch):
        """
        Run each component of the module with the given batch. The outputs are stored
        in a dictionary. If the output of a component is a dictionary, its keys are
        used as keys in the output dictionary. Otherwise, the component name is used
        as key.
        """
        out = dict()
        for name, component in module.items():
            component_out = component.run(batch)
            if isinstance(component_out, dict):
                out.update(component_out)
            else:
                out[name] = component_out
        return out

    def set_consistent_targets(self, consistent_targets):
        """
        Set whether the selected targets should be consistent. We assume that targets
        are selected in the `featproc` module, and iterate over its components looking
        for the equivalent method.
        """
        if self.featproc is None:
            LOGGER.warning("No featproc module found, so no targets to be consistent")
            return
        for name, component in self.featproc.items():
            if hasattr(component, "target_selection"):
                LOGGER.info(f"Set consistent targets of {name} to {consistent_targets}")
                component.target_selection.set_consistent_targets(consistent_targets)

    def to(self, device):
        """
        Overwrite of PyTorch's method, to propagate it to all components.
        """
        LOGGER.info(f"Changing model's device to {device}")
        for attr in [self.featex, self.featproc, self.featfusion]:
            if attr is not None:
                for component in attr.values():
                    component.to(device)
        self.synthesis.to(device)
        self.device = device
