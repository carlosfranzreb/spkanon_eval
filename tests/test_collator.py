import unittest
import torch

from spkanon_eval.datamodules.collator import collate_fn


class TestCollateFunction(unittest.TestCase):
    def test_collate_fn(self):
        # Sample input data
        batch = [
            (torch.tensor([1, 2, 3]), 0),
            (torch.tensor([4, 5]), 1),
            (torch.tensor([6, 7, 8, 9]), 0),
        ]

        # Expected output
        expected_audio = torch.tensor(
            [
                [1, 2, 3, 0],
                [4, 5, 0, 0],
                [6, 7, 8, 9],
            ]
        )
        expected_speakers = torch.tensor([0, 1, 0])

        # Call the collate function
        audio, speakers = collate_fn(batch)

        # Check if the outputs match the expected ones
        self.assertTrue(torch.equal(audio, expected_audio))
        self.assertTrue(torch.equal(speakers, expected_speakers))
