"""Tests for label resolution.

The rule these pin: a name comes from the caller or from the weights, never from a default.
Where nothing supplies one, detections carry the id alone -- which is worse than a name and far
better than a wrong one.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def labels(weights):
    from mozo import labels as module
    return module


class TestPrecedence:
    def test_caller_wins(self, labels):
        assert labels.resolve("toy", "alpha", caller=["hardhat", "vest"]) == ["hardhat", "vest"]

    def test_caller_beats_the_checkpoint(self, labels):
        assert labels.resolve("toy", "alpha", caller=["a"], checkpoint=["b"]) == ["a"]

    def test_checkpoint_beats_published(self, labels):
        assert labels.resolve("toy", "alpha", checkpoint=["mine"], published=True) == ["mine"]

    def test_falls_back_to_the_published_vocabulary(self, labels):
        assert labels.resolve("toy", "alpha", published=True) == [{"id": 1, "name": "cat"}, {"id": 5, "name": "dog"}]


class TestNoGuessing:
    def test_returns_none_when_the_model_publishes_no_labels(self, labels):
        assert labels.resolve("toy", "beta", published=True) is None

    def test_returns_none_for_an_unpublished_model(self, labels):
        assert labels.resolve("toy", "gamma", published=True) is None

    def test_returns_none_for_a_revision_without_labels(self, labels):
        assert labels.resolve("toy", "alpha", revision="2026-01-01", published=True) is None

    def test_published_labels_are_opt_in(self, labels):
        """Forgetting the flag withholds a name rather than inventing one."""
        assert labels.resolve("toy", "alpha") is None

    def test_an_empty_checkpoint_list_does_not_win(self, labels):
        """A checkpoint that carried no names must fall through, not report nothing."""
        assert labels.resolve("toy", "alpha", checkpoint=[], published=True) is not None
