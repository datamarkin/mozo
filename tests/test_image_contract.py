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
from mozo.image import load_image
from mozo.weights import WeightsError

from conftest import FIXTURE, ROOT

#: One adapter per output shape -- an array and a Detections -- since what is being pinned is
#: that the entry points agree, not that every variant does.
ADAPTERS = {"rfdetr": (RFDETRPredictor, "nano"), "depth_anything_v2": (DepthAnythingV2Predictor, "small")}

#: Function names that turn bytes or a path into pixels, and therefore decide channel order.
#: Matched on the final name, so an alias or a ``from cv2 import imdecode`` is caught too.
DECODERS = {"imread", "imdecode", "decode_image", "decode_jpeg", "read_image"}

#: The other half of the same boundary. Channel order is created at decode and destroyed at
#: encode, so an encoder written by hand is the same failure as a decoder written by hand -- and
#: the one that gets forgotten is the RGB-to-BGR step, which produces a plausible-looking picture
#: with its channels swapped.
#: ``pf.encode_image`` is written with its receiver because the name is not unique: CLIP and
#: SigLIP 2 both have an ``encode_image`` that produces an embedding, which is a different
#: operation that happens to share a word. Same reason ``Image.open`` is matched on both halves.
ENCODERS = {"imencode", "imwrite", "pf.encode_image"}

#: mozo.image is where the contract is declared; vendored code keeps its upstream's own.
#: mozo.depth encodes too, and is exempt only for encoding: a depth map is 16-bit single-channel,
#: so it has no channel order to get wrong, and its endpoints have to travel with it.
EXEMPT = {ROOT / "mozo" / "image.py"}
EXEMPT_FROM_ENCODING = EXEMPT | {ROOT / "mozo" / "depth.py"}


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
        # Channel order is the one thing an ndarray cannot be asked about, so an RGB uint8 array
        # is returned untouched rather than guessed at. Anything else would be inventing a fact
        # about the caller's data.
        array = np.zeros((4, 4, 3), dtype=np.uint8)
        assert load_image(array) is array

    @pytest.mark.parametrize("array, why", [
        (np.zeros((4, 4), np.uint8), "grayscale"),
        (np.zeros((4, 4, 4), np.uint8), "RGBA"),
        (np.zeros((4, 4, 3), np.float32), "float"),
        (np.zeros((1, 4, 4, 3), np.uint8), "four dimensions"),
    ])
    def test_an_array_of_the_wrong_kind_is_refused(self, array, why):
        """Trusted on channel order is not trusted on everything else.

        Order cannot be recovered from pixels, so it is taken on the caller's word. Shape and
        dtype can, so they are checked -- and a grayscale or float array stops here rather than
        reaching a model that will read it as something it is not.
        """
        with pytest.raises(ValueError):
            load_image(array)

    def test_a_missing_path_is_an_error_not_a_none(self):
        with pytest.raises(FileNotFoundError):
            load_image("does/not/exist.jpg")

    def test_undecodable_bytes_raise(self):
        with pytest.raises(ValueError):
            load_image(b"not an image")

    def test_an_unsupported_type_names_what_is_accepted(self):
        """``TypeError``, because 42 is the wrong kind of thing rather than a bad value of the
        right kind -- which is the distinction undecodable bytes are on the other side of.

        Matched on what the message has to say, not on how it says it: the wording belongs to
        PixelFlow now, and rewording it there should not fail a test here.
        """
        with pytest.raises(TypeError, match=r"path.*buffer.*array"):
            load_image(42)


#: Names that mean one thing on their own and another with a receiver in front. ``Image.open``
#: decodes where ``target.open("wb")`` writes a file; ``pf.encode_image`` encodes where
#: ``self._encoder.encode_image`` embeds. Matched on both halves so the two do not collide.
QUALIFIED = {"open", "encode_image"}


def called(node: ast.Call) -> str:
    """The name being called, however it was imported or aliased.

    Shared by both sweeps rather than nested in one, because "what is being called" is the same
    question either way and a second copy is how the two rules come to disagree about it.
    """
    if isinstance(node.func, ast.Attribute):
        receiver = getattr(node.func.value, "id", "")
        return (f"{receiver}.{node.func.attr}" if node.func.attr in QUALIFIED
                else node.func.attr)
    return getattr(node.func, "id", "")


class TestOneEncodeBoundary:
    def test_nothing_outside_image_py_encodes_an_image(self):
        """The mirror of the decode rule, and it exists because the substring version failed.

        The first attempt asserted that ``mozo/workflow/api.py`` no longer contained the string
        ``def _png``. It passed while ``mozo/server.py`` imported that very function from that very
        module and raised ``ImportError`` on every call, because the property is repo-wide and a
        substring search of one file cannot hold a repo-wide property. This is the same sweep the
        decode rule uses, over the calls that turn an array into bytes.
        """
        scanned, offenders = 0, []
        for path in sorted((ROOT / "mozo").rglob("*.py")):
            if path in EXEMPT_FROM_ENCODING or "vendors" in path.parts:
                continue
            scanned += 1
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and (name := called(node)) in ENCODERS:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")

        assert scanned > 5, f"only scanned {scanned} files under {ROOT / 'mozo'}"
        assert not offenders, (
            "these encode an image without going through mozo.image.encode_image, which is where "
            f"RGB-to-BGR is decided: {offenders}")


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
            "these decode images outside mozo.image.load_image, so mozo would have more than "
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


class TestTheAlphaChannel:
    """``encode_image`` takes three channels or four, because a cut-out is a picture.

    The one-encoder rule exists because a hand-written encoder forgets the RGB-to-BGR swap and
    produces a plausible picture with its channels wrong -- the failure that cost Depth Anything
    V2 0.166 m of mean error. Widening that function to four channels widens the surface the rule
    protects, so the swap is checked on the wider path too.
    """

    def test_rgba_survives_the_round_trip_in_the_right_order(self):
        import io

        from PIL import Image as PILImage

        from mozo.image import encode_image

        # Three different colour values, so a swap cannot hide behind a symmetry.
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 10, 120, 230, 200

        decoded = np.array(PILImage.open(io.BytesIO(encode_image(rgba))))
        assert decoded.shape == (4, 4, 4)
        assert tuple(decoded[0, 0]) == (10, 120, 230, 200), "channels came back swapped"

    def test_the_alpha_is_lossless(self):
        import io

        from PIL import Image as PILImage

        from mozo.image import encode_image

        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., 3] = np.arange(64, dtype=np.uint8).reshape(8, 8)
        decoded = np.array(PILImage.open(io.BytesIO(encode_image(rgba))))
        # A matte that came back approximately is not a matte.
        assert np.array_equal(decoded[..., 3], rgba[..., 3])

    def test_three_channels_still_go_through_pixelflow_unchanged(self):
        import io

        from PIL import Image as PILImage

        from mozo.image import encode_image

        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 120, 230
        decoded = np.array(PILImage.open(io.BytesIO(encode_image(rgb))))
        assert decoded.shape == (4, 4, 3)
        assert tuple(decoded[0, 0]) == (10, 120, 230)

    def test_a_format_that_cannot_carry_alpha_is_refused_not_flattened(self):
        """Flattening picks a background colour for the caller. That is a decision, not a
        conversion, so it is refused by name instead."""
        from mozo.image import encode_image

        with pytest.raises(ValueError, match=r"\.jpg cannot carry an alpha channel"):
            encode_image(np.zeros((4, 4, 4), dtype=np.uint8), ".jpg")

    def test_as_rgb_drops_alpha_and_leaves_rgb_alone(self):
        from mozo.image import as_rgb

        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[..., 3] = 200
        assert as_rgb(rgba).shape == (4, 4, 3)

        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        assert as_rgb(rgb) is rgb
