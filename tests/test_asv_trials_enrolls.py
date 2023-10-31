import os
import json
import unittest
import shutil

from spkanon_eval.evaluation.asv.trials_enrolls import split_trials_enrolls


class TestTrialsEnrolls(unittest.TestCase):
    def setUp(self):
        self.exp_folder = "spkanon_eval/tests/logs/split_trials_enrolls"
        if os.path.isdir(self.exp_folder):
            shutil.rmtree(self.exp_folder)
        os.makedirs(os.path.join(self.exp_folder, "data"))
        self.root_folder = "tests/data"
        datafile = "spkanon_eval/tests/datafiles/ls-dev-clean-2.txt"
        shutil.copy(datafile, os.path.join(self.exp_folder, "data", "eval.txt"))
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

        f_trials, f_enrolls = split_trials_enrolls(self.exp_folder, self.root_folder)

        # Check that the files were created
        self.assertTrue(os.path.isfile(f_trials))
        self.assertTrue(os.path.isfile(f_enrolls))

        # Check the content of the files
        trial_fpaths = [
            os.path.split(json.loads(line)["path"])[1] for line in open(f_trials)
        ]
        enroll_fpaths = [
            os.path.split(json.loads(line)["path"])[1] for line in open(f_enrolls)
        ]
        self.assertCountEqual(trial_fpaths, self.expected_trials)
        self.assertCountEqual(enroll_fpaths, self.expected_enrolls)

    def tearDown(self):
        """Remove the created directory"""
        shutil.rmtree(self.exp_folder)


if __name__ == "__main__":
    unittest.main()
