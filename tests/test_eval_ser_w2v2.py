import os
from base import BaseTestClass, run_pipeline
import torch
from omegaconf import OmegaConf

from spkanon_eval.evaluation.ser.audeering_w2v import EmotionEvaluator


class TestEvalSer(BaseTestClass):
    def test_results(self):
        """
        Test whether the SER results match the expected values.
        """

        # run the experiment with both ASV evaluation scenarios
        self.init_config.eval.components = {
            "audeering_w2v": {
                "cls": "spkanon_eval.evaluation.ser.audeering_w2v.EmotionEvaluator",
                "init": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                "train": False,
                "num_workers": 0,
                "batch_size": 2,
            }
        }
        self.config, self.log_dir = run_pipeline(self.init_config)

        # get the directory where the results are stored
        results_subdir = "eval/ser-audeering-w2v"
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

    def test_batch(self):
        """
        Test whether the batching works: the results of the batch should equal the results of the
        individual samples. We check the emotion embeddings for each sample.
        """
        audios = torch.randn(4, 50000) - 0.5
        config = OmegaConf.create(
            {
                "init": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                "batch_size": 2,
                "data": {
                    "config": {
                        "trainer": {"batch_size": 2},
                        "sample_rate": 16000,
                    },
                },
            }
        )
        evaluator = EmotionEvaluator(config, "cpu")
        batched_out = evaluator.run([audios])[0]
        single_out = [evaluator.run([audio.unsqueeze(0)])[0] for audio in audios]
        for i in range(len(batched_out)):
            self.assertTrue(torch.allclose(batched_out[i], single_out[i]))
            if i > 0:
                self.assertFalse(torch.allclose(batched_out[i], batched_out[i - 1]))
