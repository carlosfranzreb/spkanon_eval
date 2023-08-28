"""
Test the evaluation components. We don't check whether the numbers are right, like the
EER or the LLRs, but rather that these numbers are computed for the correct speakers
and utterances. This test class inherits from BaseTestClass, which runs the inference
for the debug data.
"""


import os

from omegaconf import OmegaConf

from base import BaseTestClass, run_pipeline


class TestEvalPerformance(BaseTestClass):
    def test_results(self):
        """
        Test whether the Performance results match the expected values. We only test
        the CPU results, not the GPU results.
        """

        # run the experiment with both ASV evaluation scenarios
        self.init_config.eval.components = OmegaConf.create(
            {
                "performance": {
                    "cls": "spkanon_eval.evaluation.performance.performance.PerformanceEvaluator",
                    "train": False,
                    "repetitions": 2,
                    "sample_rate": "${sample_rate}",
                    "durations": [2, 4],
                },
            }
        )
        self.config, self.log_dir = run_pipeline(self.init_config)

        # get the directory where the results are stored
        results_subdir = "eval/performance"
        results_dir = os.path.join(self.log_dir, results_subdir)
        expected_dir = os.path.join("tests", "expected_results", results_subdir)

        # assert that both directories contain the same files
        self.assertCountEqual(os.listdir(results_dir), os.listdir(expected_dir))

        # assert that the results files contain the same lines
        for fname in os.listdir(results_dir):
            with open(os.path.join(results_dir, fname)) as f:
                results = f.readlines()

            # for `cpu_specs.txt`, compare it with the CPU in this machine
            if fname == "cpu_specs.txt":
                f_expected = os.path.join(expected_dir, fname)
                os.system(f"sysctl -a | grep machdep.cpu > {f_expected}")
                with open(os.path.join(expected_dir, fname)) as f:
                    expected = f.readlines()
                with self.subTest(fname=fname):
                    self.assertEqual(results, expected)

            # check that the header and first col match, but ignore the numbers
            else:
                with open(os.path.join(expected_dir, fname)) as f:
                    expected = f.readlines()
                with self.subTest(fname=fname):
                    self.assertEqual(results[0], expected[0])
                    self.assertEqual(
                        [line.split()[0] for line in results],
                        [line.split()[0] for line in expected],
                    )
