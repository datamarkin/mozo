"""One colour order, one decode boundary, for every way an image can enter mozo.

Channel order is created when bytes are decoded and invisible afterwards -- a numpy array
carries no colour metadata, so nothing downstream can tell RGB from BGR. That makes this a
contract rather than an implementation detail, and one that fails silently when broken: feeding
a model the wrong channel order does not raise, it just returns a slightly wrong answer. Depth
Anything V2 was measured at 0.166 m mean error, 1.84 m worst, from a channel swap alone.

So these tests exist to make the three entry points provably agree, and to keep the number of
places that decode at one.
"""

from __future__ import annotations

import ast

import cv2
import numpy as np
import pytest

from mozo.adapters.depth_anything_v2 import DepthAnythingV2Predictor
from mozo.adapters.rfdetr import RFDETRPredictor
from mozo.utils import load_image
from mozo.weights import WeightsError

from conftest import FIXTURE, ROOT

#: One adapter per output shape -- an array and a Detections -- since what is being pinned is
#: that the entry points agree, not that every variant does.
ADAPTERS = {"rfdetr": (RFDETRPredictor, "nano"), "depth_anything_v2": (DepthAnythingV2Predictor, "small")}

#: Function names that turn bytes or a path into pixels, and therefore decide channel order.
#: Matched on the final name, so an alias or a ``from cv2 import imdecode`` is caught too.
DECODERS = {"imread", "imdecode", "decode_image", "decode_jpeg", "read_image"}

#: mozo.utils is where the contract is declared; vendored code keeps its upstream's own.
EXEMPT = {ROOT / "mozo" / "utils.py"}


@pytest.fixture(scope="module")
def predictors():
    built = {}
    for family, (cls, variant) in ADAPTERS.items():
        try:
            built[family] = cls(variant, device="cpu")
        except WeightsError as error:
            pytest.skip(f"{family} weights unavailable: {error}")
    return built


@pytest.fixture(scope="module")
def baseline(predictors, image):
    """Each family's answer for the fixture, computed once and reused."""
    return {family: model.predict(image) for family, model in predictors.items()}


def same(a, b) -> bool:
    """Compare two predictions of whichever kind the family returns."""
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a.to_dict() == b.to_dict()


class TestLoadImage:
    def test_a_path_decodes_to_rgb(self):
        assert np.array_equal(load_image(FIXTURE), cv2.imread(FIXTURE)[..., ::-1])

    def test_bytes_decode_to_the_same_pixels_as_the_path(self, payload):
        assert np.array_equal(load_image(payload), load_image(FIXTURE))

    def test_an_array_is_taken_at_its_word(self):
        # There is nothing in an ndarray to inspect, so it is returned untouched rather than
        # guessed at. Anything else would be inventing a fact about the caller's data.
        array = np.zeros((4, 4, 3), dtype=np.uint8)
        assert load_image(array) is array

    def test_a_missing_path_is_an_error_not_a_none(self):
        with pytest.raises(FileNotFoundError):
            load_image("does/not/exist.jpg")

    def test_undecodable_bytes_raise(self):
        with pytest.raises(ValueError):
            load_image(b"not an image")

    def test_an_unsupported_type_names_what_is_accepted(self):
        with pytest.raises(ValueError, match="path .str or Path., encoded bytes, or numpy array"):
            load_image(42)


class TestOneDecodeBoundary:
    def test_nothing_outside_load_image_decodes_an_image(self):
        """Every decoder call in the package, found by parsing rather than by grepping.

        The property is repo-wide -- "channel order is decided in exactly one place" -- so
        asserting that one file does not contain one substring would not test it. That version
        passed if the server switched to ``PIL.Image.open`` and failed if someone wrote the
        word in a comment.

        Verified to catch ``cv2.imread``, ``cv2.imdecode``, ``Image.open``, and a bare
        ``imdecode`` from a ``from cv2 import ...``. It does not catch a *renamed* import
        (``from cv2 import imdecode as _d``), which is a deliberate disguise rather than the
        accident this guards against -- resolving those needs an alias map that buys nothing
        else.
        """
        def called(node: ast.Call) -> str:
            """The name being called, however it was imported or aliased."""
            if isinstance(node.func, ast.Attribute):
                receiver = getattr(node.func.value, "id", "")
                # ``Image.open`` decodes; ``target.open("wb")`` writes a file. Only the
                # receiver tells them apart, so that one pair is matched on both halves.
                return f"{receiver}.{node.func.attr}" if node.func.attr == "open" else node.func.attr
            return getattr(node.func, "id", "")

        scanned, offenders = 0, []
        for path in sorted((ROOT / "mozo").rglob("*.py")):
            if path in EXEMPT or "vendors" in path.parts:
                continue
            scanned += 1
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call):
                    name = called(node)
                    if name in DECODERS or name == "Image.open":
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")

        # An empty sweep -- wrong ROOT, package moved, run against an installed wheel -- would
        # otherwise pass green while checking nothing.
        assert scanned > 5, f"only scanned {scanned} files under {ROOT / 'mozo'}"
        assert not offenders, (
            "these decode images outside mozo.utils.load_image, so mozo would have more than "
            f"one place deciding channel order: {offenders}"
        )


class TestEveryEntryPointAgrees:
    """A file, an array from PixelFlow, and a request body must give one answer."""

    def test_pixelflow_hands_over_exactly_what_load_image_would(self):
        """The reason mozo chose RGB: its companion library already had.

        Asserted on the pixels rather than by running a model on them -- feeding two identical
        arrays to a deterministic model and comparing the outputs proves nothing the array
        comparison does not, and costs an inference per family.
        """
        import pixelflow as pf

        assert np.array_equal(pf.read_image(FIXTURE), load_image(FIXTURE))

    @pytest.mark.parametrize("family", list(ADAPTERS))
    def test_path_array_and_bytes_are_interchangeable(self, predictors, baseline, payload, family):
        model = predictors[family]
        for other in (model.predict(FIXTURE), model.predict(payload)):
            assert same(baseline[family], other)

    @pytest.mark.parametrize("family", list(ADAPTERS))
    def test_feeding_bgr_actually_changes_the_answer(self, predictors, baseline, image, family):
        """Guards the guard: if RGB and BGR gave the same result, the tests above prove nothing."""
        swapped = predictors[family].predict(np.ascontiguousarray(image[..., ::-1]))
        assert not same(baseline[family], swapped)
