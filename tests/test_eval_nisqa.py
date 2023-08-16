"""
Test the evaluation components. We don't check whether the numbers are right, like the
EER or the LLRs, but rather that these numbers are computed for the correct speakers
and utterances. This test class inherits from BaseTestClass, which runs the inference
for the debug data.
"""


import pickle
from string import Template
import json
import os

from base import BaseTestClass, run_pipeline
from src.setup_module import setup


NISQA_CFG = {
    "naturalness_nisqa": {
        "cls": "src.evaluation.naturalness.naturalness_nisqa.NisqaEvaluator",
        "init": "submodules/NISQA/weights/nisqa_tts.tar",
        "train": False,
        "num_workers": "${trainer.num_workers}",
        "batch_size": 10,
    }
}


class TestEvalNisqa(BaseTestClass):
    def test_nisqa(self):
        """
        Check that NISQA outputs MOS scores for all utterances in the test set,
        regardless what the scores are. Also, to ensure that the scores and filenames
        are properly matched, compute again the score of the minimum and maximum scores
        and test whether the new scores are the same as the old ones (aprox).
        """

        # run the experiment with both ASV evaluation scenarios
        self.init_config.eval.components = NISQA_CFG
        self.config, self.log_dir = run_pipeline(self.init_config)

        # assert that the directory exists
        expected_dir = os.path.join(self.log_dir, "eval", "nisqa")
        self.assertTrue(os.path.isdir(expected_dir))

        # assert that all eval utterances are included in the output
        extrema = dict()
        for eval_file in self.config.data.datasets.eval:
            # gather the utterances from this eval file
            expected_utts = list()
            for line in open(os.path.join(self.log_dir, "results", eval_file)):
                expected_utts.append(json.loads(line)["audio_filepath"])

            # gather the utts from the NISQA output and ensure the scores are valid
            fname = os.path.basename(eval_file)
            output_file = os.path.join(expected_dir, fname)
            eval_utts, eval_scores = list(), list()
            with open(output_file) as f:
                next(f)  # skip header
                for line in f:
                    utt, score = line.split()
                    self.assertGreaterEqual(float(score), 0)
                    self.assertLessEqual(float(score), 5)
                    eval_utts.append(utt)
                    eval_scores.append(float(score))

            # assert that all utts are included in the output
            self.assertCountEqual(expected_utts, eval_utts)

            # add fnames of extrema scores to a new datafile
            extrema[fname] = dict()
            extrema[fname]["min"] = min(eval_scores)
            extrema[fname]["max"] = max(eval_scores)
            min_idx = eval_scores.index(extrema[fname]["min"])
            max_idx = eval_scores.index(extrema[fname]["max"])
            datafile = os.path.join(self.log_dir, "results", f"extrema_{fname}")
            with open(datafile, "w") as f:
                f.write(json.dumps({"audio_filepath": eval_utts[min_idx]}))
                f.write(json.dumps({"audio_filepath": eval_utts[max_idx]}))

        # run evaluation again and check the extrema scores
        nisqa = setup(self.config.eval.components.naturalness_nisqa, "cpu")
        datafiles = [
            os.path.join(self.log_dir, "results", f)
            for f in self.config.data.datasets.eval
        ]
        nisqa.eval_dir(self.log_dir, datafiles, False)
        for eval_file in os.listdir(expected_dir):
            if not eval_file.startswith("extrema_"):
                continue
            fname = eval_file[len("extrema_") :]
            for i, line in enumerate(open(os.path.join(expected_dir, eval_file))):
                utt, score = line.split()
                if i == 0:
                    self.assertAlmostEqual(
                        float(score), extrema[fname]["min"], places=1
                    )
                elif i == 1:
                    self.assertAlmostEqual(
                        float(score), extrema[fname]["max"], places=1
                    )

    def test_results(self):
        """
        Compare the results with those that we have checked manually.
        """

        # run the experiment with both ASV evaluation scenarios
        self.init_config.eval.components = NISQA_CFG
        self.config, self.log_dir = run_pipeline(self.init_config)

        # get the directory where the results are stored
        results_subdir = "eval/nisqa"
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
