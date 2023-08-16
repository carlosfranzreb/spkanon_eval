import os
import json
import unittest
import shutil

from omegaconf import OmegaConf

from src.evaluation.asv.asv_utils import filter_samples


class TestFilterSamples(unittest.TestCase):
    def setUp(self):
        self.datafile = "data/debug/ls-dev-clean-2.txt"
        self.expected_out_file = "data/debug/asv_splits/ls-dev-clean-2_filtered.txt"
        os.makedirs(os.path.dirname(self.expected_out_file), exist_ok=True)

    def test_filter_samples(self):
        """Test the function with two different configurations"""

        filter_cfg = OmegaConf.create({"min_samples": 2, "min_dur": 4})
        out_file, n_speakers = filter_samples(self.datafile, filter_cfg)
        expected_fnames = [
            "2412-153948-0000.flac",
            "2412-153948-0001.flac",
            "1988-24833-0001.flac",
            "1988-24833-0002.flac",
            "1988-24833-0003.flac",
        ]
        with open(self.expected_out_file) as f:
            for line in f:
                obj = json.loads(line.strip())
                fname = os.path.split(obj["audio_filepath"])[1]
                self.assertIn(fname, expected_fnames)

        filter_cfg = OmegaConf.create({"min_samples": 3, "min_dur": 4.5})
        out_file, n_speakers = filter_samples(self.datafile, filter_cfg)
        expected_fnames = [
            "1988-24833-0001.flac",
            "1988-24833-0002.flac",
            "1988-24833-0003.flac",
        ]
        with open(self.expected_out_file) as f:
            for line in f:
                obj = json.loads(line.strip())
                fname = os.path.split(obj["audio_filepath"])[1]
                self.assertIn(fname, expected_fnames)

    def tearDown(self):
        """Remove the created directory"""
        shutil.rmtree(os.path.dirname(self.expected_out_file))


if __name__ == "__main__":
    unittest.main()
