"""
Test the evaluation components. We don't check whether the numbers are right, like the
EER or the LLRs, but rather that these numbers are computed for the correct speakers
and utterances. This test class inherits from BaseTestClass, which runs the inference
for the debug data.
"""


import os

from base import BaseTestClass, run_pipeline


class TestEvalWhisper(BaseTestClass):
    def test_results(self):
        """
        Test whether the ASR results match the expected values.
        """

        # run the experiment with both ASV evaluation scenarios
        self.init_config.eval.components = {
            "whisper_tiny": {
                "cls": "spkanon_eval.featex.asr.whisper.Whisper",
                "train": False,
                "size": "tiny",
                "output": "text",
                "batch_size": 16,
            }
        }
        self.config, self.log_dir = run_pipeline(self.init_config)

        # get the directory where the results are stored
        results_subdir = "eval/whisper-tiny"
        results_dir = os.path.join(self.log_dir, results_subdir)
        expected_dir = os.path.join("tests", "expected_results", results_subdir)

        # assert that both directories contain the same files
        self.assertCountEqual(os.listdir(results_dir), os.listdir(expected_dir))

        # assert that the results files contain the same lines
        for fname in os.listdir(results_dir):
            # check that the results file is the same as the expected results file
            with open(os.path.join(results_dir, fname)) as f:
                results = f.readlines()
            with open(os.path.join(expected_dir, fname)) as f:
                expected = f.readlines()

            # if the file contains the results for each utterance, ignore the fnames
            if results[0].startswith("audio_filepath"):
                results = [r.split()[1:] for r in results]
                expected = [e.split()[1:] for e in expected]

            with self.subTest(fname=fname):
                self.assertCountEqual(results, expected)
