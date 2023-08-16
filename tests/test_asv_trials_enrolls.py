import os
import json
import unittest
import shutil

from src.evaluation.asv.trials_enrolls import split_trials_enrolls


class TestTrialsEnrolls(unittest.TestCase):
    def setUp(self):
        self.datafile = "data/debug/ls-dev-clean-2.txt"
        self.f_trials = "data/debug/asv_splits/ls-dev-clean-2_trials.txt"
        self.f_enrolls = "data/debug/asv_splits/ls-dev-clean-2_enrolls.txt"
        self.expected_trials = [
            "2412-153948-0000.flac",
            "3752-4944-0000.flac",
            "1988-24833-0000.flac",
        ]
        self.expected_enrolls = [
            "2412-153948-0001.flac",
            "2412-153948-0002.flac",
            "3752-4944-0001.flac",
            "3752-4944-0002.flac",
            "1988-24833-0001.flac",
            "1988-24833-0002.flac",
            "1988-24833-0003.flac",
        ]

    def test_split(self):
        """Test that the splits work, without testing the `set_anon_path` function. We
        therefore set `is_baseline` to True, which will return the original object."""

        f_trials, f_enrolls = split_trials_enrolls("", self.datafile, "", True)

        # Check that the files were created
        self.assertTrue(os.path.isfile(f_trials))
        self.assertTrue(os.path.isfile(f_enrolls))

        # Check that the files are in the expected locations
        self.assertEqual(f_trials, self.f_trials)
        self.assertEqual(f_enrolls, self.f_enrolls)

        # Check the content of the files
        trial_fpaths = [
            os.path.split(json.loads(line)["audio_filepath"])[1]
            for line in open(f_trials)
        ]
        enroll_fpaths = [
            os.path.split(json.loads(line)["audio_filepath"])[1]
            for line in open(f_enrolls)
        ]
        self.assertCountEqual(trial_fpaths, self.expected_trials)
        self.assertCountEqual(enroll_fpaths, self.expected_enrolls)

    def tearDown(self):
        """Remove the created directory"""
        shutil.rmtree(os.path.dirname(self.f_trials))


if __name__ == "__main__":
    unittest.main()
