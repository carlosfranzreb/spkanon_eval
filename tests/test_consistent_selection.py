"""
Test whether the targets are consistent across the utterances of each speaker.
The base target selector is reponsible for this.
"""

import unittest
import torch
from omegaconf import OmegaConf

from src.target_selection.random import RandomSelector


class TestConsistentSelection(unittest.TestCase):
    def test_consistent_selection(self):
        # ensure that given targets are propagated to all utts of the same speaker
        cfg = OmegaConf.create({"consistent_targets": True})
        selector = RandomSelector(torch.randn(20, 16), cfg)
        source = ["a", "a", "b", "b", "b", "c"]
        target_in = torch.tensor([-1, -1, 0, -1, -1, 2])
        spec = torch.randn(6, 80, 100)
        target_1 = selector.select(spec, source, target_in)
        self.assertTrue(target_1[0] == target_1[1])
        self.assertTrue(torch.all(target_1[2:5] == 0))
        self.assertTrue(target_1[5] == 2)

        # ensure that targets defined in previous calls are used again
        source = ["a", "b", "c", "d"]
        spec = torch.randn(4, 80, 100)
        target_2 = selector.select(spec, source)
        self.assertTrue(target_2[:3].tolist() == [target_1[0].item(), 0, 2])

    def test_inconsistent_selection(self):
        # ensure that targets change between calls
        cfg = OmegaConf.create({"consistent_targets": False})
        selector = RandomSelector(torch.randn(1000, 16), cfg)
        source = ["a", "a", "b", "b", "b", "c"]
        spec = torch.randn(6, 80, 100)
        target_1 = selector.select(spec, source)
        target_2 = selector.select(spec, source)
        self.assertFalse(torch.all(target_1 == target_2))

        # ensure that given targets are kept
        target_in = torch.tensor([-1, -1, 0, -1, -1, 2])
        target_3 = selector.select(spec, source, target_in)
        self.assertTrue(target_3[2] == 0)
        self.assertTrue(target_3[5] == 2)
