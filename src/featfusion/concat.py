import torch


class Concat:
    def __init__(self, config):
        self.timed = config["time_dependent"]
        self.nontimed = config["time_invariant"]
        if isinstance(self.timed, str):
            self.timed = [self.timed]
        if isinstance(self.nontimed, str):
            self.nontimed = [self.nontimed]
    
    def run(self, feats):
        concat = torch.empty(self._get_dims(feats))
        start = 0
        for feat in self.timed:
            data = feats[feat].transpose(1, 2)
            concat[:, start:data.shape[1]+start, :] = data
            start += data.shape[1]
        for feat in self.nontimed:
            data = feats[feat].unsqueeze(2)
            concat[:, start:data.shape[1]+start, :] = data
            start += data.shape[1]
        return concat

    def _get_dims(self, feats):
        """Return the dims of the concatenated feats. Raise an error if the dims of
        the feats don't match, if the no. of batches differ among feats, or if a
        feat is not in the lists of expected feats."""
        batch, time, channels = None, None, 0
        for name, feat in feats.items():
            if batch is None:
                batch = feat.shape[0]
            elif feat.shape[0] != batch:
                raise ValueError("Feats have different n_batches")
            if name in self.timed:
                assert len(feat.shape) == 3  # batch x dims x time
                if time is None:
                    time = feat.shape[1]
                elif feat.shape[1] != time:
                    raise ValueError("Timed feats have different n_dims")
                channels += feat.shape[2]
            elif name in self.nontimed:
                assert len(feat.shape) == 2  # batch x dims
                channels += feat.shape[1]
            else:
                raise ValueError(f"{name} feat not expected")
        return [batch, channels, time]
