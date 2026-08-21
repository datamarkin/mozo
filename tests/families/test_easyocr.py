"""EasyOCR: the two graphs' shapes, the preprocessing traps, and the CTC alphabet.

Bit-exactness against the published package lives in ``tools/verify/easyocr.py``, which needs
the weights. What this file pins is everything that would still be green if the model had
quietly changed: the alphabet layout that decides whether a string decodes shifted, the
preprocessing details that are individually invisible and collectively the difference between
reading a sign and inventing one, and the two places where a plausible-looking simplification
changes the output.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from conftest import TEXT_FIXTURES, require_weights
from mozo.registry import get_model_info
from mozo.vendors.easyocr_deploy import boxes, image
from mozo.vendors.easyocr_deploy.config import SPECS, VARIANTS
from mozo.vendors.easyocr_deploy.craft import CRAFT
from mozo.vendors.easyocr_deploy.crnn import CRNN
from mozo.vendors.easyocr_deploy.text import BLANK, Alphabet

FAMILY = "easyocr"


# --- the alphabet -------------------------------------------------------------------------------

def test_the_blank_takes_index_zero_and_shifts_every_character_up():
    """CTC needs a blank and it is index 0, so a character's index is its position in the
    published charset plus one. Off by one here shifts every string the model produces."""
    alphabet = Alphabet("abc")
    assert BLANK == 0
    assert alphabet.characters == ["[blank]", "a", "b", "c"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_num_class_is_the_charset_plus_the_blank(variant):
    """The recogniser's final linear layer is this wide, so a charset that disagrees with the
    checkpoint is a strict load that cannot succeed."""
    spec = SPECS[variant]
    assert spec.num_class == len(spec.characters) + 1
    assert len(Alphabet(spec.characters).characters) == spec.num_class


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_charset_repeats_a_character(variant):
    """A duplicate would make two indices decode the same glyph, and the second unreachable."""
    characters = SPECS[variant].characters
    assert len(characters) == len(set(characters))


def test_repeated_steps_collapse_but_a_blank_between_them_does_not():
    """This is the whole of CTC. ``a a`` is one 'a'; ``a <blank> a`` is two."""
    alphabet = Alphabet("ab")
    one = torch.full((1, 3, 3), -20.0)
    one[0, 0, 1] = one[0, 1, 1] = one[0, 2, 1] = 20.0        # a a a
    assert alphabet.decode(one)[0][0] == "a"

    two = torch.full((1, 3, 3), -20.0)
    two[0, 0, 1] = two[0, 2, 1] = 20.0                        # a _ a
    two[0, 1, BLANK] = 20.0
    assert alphabet.decode(two)[0][0] == "aa"


def test_an_all_blank_row_reads_as_nothing_with_zero_confidence():
    """A located region the recogniser had nothing for. It is kept rather than dropped, and its
    confidence is 0 rather than a fabricated number."""
    alphabet = Alphabet("ab")
    logits = torch.full((1, 4, 3), -20.0)
    logits[0, :, BLANK] = 20.0
    text, confidence = alphabet.decode(logits)[0]
    assert text == ""
    assert confidence == 0.0


def test_confidence_is_not_a_mean():
    """It is ``prod ** (2 / sqrt(n))``. The distinction matters because this is also the number
    the low-contrast retry compares to decide which of two reads to keep."""
    alphabet = Alphabet("ab")
    logits = torch.zeros(1, 4, 3)
    logits[0, :, 1] = 10.0
    _, confidence = alphabet.decode(logits)[0]
    probabilities = torch.softmax(logits, dim=2)[0, :, 1].numpy()
    assert confidence == pytest.approx(
        float(probabilities.prod() ** (2.0 / np.sqrt(len(probabilities)))))
    assert confidence != pytest.approx(float(probabilities.mean()))


# --- preprocessing ------------------------------------------------------------------------------

def test_the_crop_resize_is_bilinear_despite_saying_lanczos_upstream():
    """Upstream passes ``Image.Resampling.LANCZOS`` to ``cv2.resize``. PIL's LANCZOS is 1 and so
    is ``cv2.INTER_LINEAR``; OpenCV's own Lanczos is 4. Matching the published model means
    keeping the filter that call actually selects, not the one it reads like."""
    assert int(cv2.INTER_LINEAR) == 1
    assert image._CROP_INTERPOLATION == cv2.INTER_LINEAR
    assert cv2.INTER_LANCZOS4 == 4


def test_padding_replicates_the_last_column_rather_than_filling_with_zeros():
    """Zero padding reads as a black bar, which the recogniser happily decodes as characters."""
    crop = np.full((64, 32), 200, dtype=np.uint8)
    crop[:, -1] = 17
    batch = image.align(crop, width=128)
    padded = batch[0, 0, :, 32:]
    assert torch.allclose(padded, torch.full_like(padded, 17 / 255.0 * 2 - 1))
    assert not torch.allclose(padded, torch.zeros_like(padded))


def test_the_detector_pads_to_a_multiple_of_thirty_two():
    """The network downsamples by 32, so a size that does not divide by it cannot forward."""
    batch, ratio = image.for_detector(np.zeros((100, 150, 3), dtype=np.uint8))
    assert batch.shape[2] % 32 == 0 and batch.shape[3] % 32 == 0
    assert ratio == 1.0


def test_a_large_image_is_capped_and_reports_the_ratio_that_undoes_it():
    """Boxes come back in network space and are multiplied by ``1 / ratio``, so a wrong ratio
    puts every box in the wrong place rather than failing."""
    batch, ratio = image.for_detector(np.zeros((4000, 2000, 3), dtype=np.uint8))
    assert ratio == pytest.approx(image.CANVAS_SIZE / 4000)
    assert max(batch.shape[2:]) <= image.CANVAS_SIZE + 31


def test_the_padded_width_is_quantised_to_whole_crop_heights():
    """A whole multiple of the model height, from this line's own aspect ratio and nothing
    else -- upstream's other path pads to the widest crop on the page instead."""
    grey = np.zeros((200, 600), dtype=np.uint8)
    _quad, _crop, width = image.line_image([10, 400, 20, 70], grey, is_free=False)
    assert width % image.MODEL_HEIGHT == 0


def test_a_degenerate_line_is_dropped_rather_than_read():
    """A zero-height slice has no crop to give the recogniser."""
    grey = np.zeros((200, 600), dtype=np.uint8)
    assert image.line_image([10, 400, 50, 50], grey, is_free=False) is None


# --- postprocessing -----------------------------------------------------------------------------

def test_the_heatmap_stride_is_undone_before_the_resize_ratio():
    """Two factors, not one: the heatmaps are half the network input, and the network input was
    the image scaled. Dropping the stride halves every box."""
    quad = [np.array([[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])]
    assert boxes.rescale(quad, 1.0)[0].tolist() == [2, 4, 6, 4, 6, 8, 2, 8]


def test_rescaling_truncates_rather_than_rounds():
    """``int32``, which is upstream's cast, and the grouping reads the truncated values."""
    quad = [np.array([[0.9, 0.9], [1.9, 0.9], [1.9, 1.9], [0.9, 1.9]])]
    assert boxes.rescale(quad, 2.0)[0].tolist() == [0, 0, 1, 0, 1, 1, 0, 1]


def test_the_thresholds_are_the_ones_readtext_uses_not_the_ones_group_text_box_defaults_to():
    """Upstream's ``group_text_box`` signature says 1.0 and 0.05; its reader passes 0.5 and 0.1.
    These decide which crops exist, so the wrong pair changes the text, not just the boxes."""
    assert boxes.WIDTH_THS == 0.5
    assert boxes.ADD_MARGIN == 0.1
    assert boxes.MIN_SIZE == 20


def test_an_empty_page_groups_to_nothing():
    assert boxes.group([]) == ([], [])


def test_a_level_quad_groups_horizontal_and_a_tilted_one_groups_free():
    """The split decides whether a line is sliced out of the page or perspective-warped."""
    level = np.array([10, 10, 200, 10, 200, 60, 10, 60], dtype=np.int32)
    tilted = np.array([10, 10, 200, 60, 190, 110, 0, 60], dtype=np.int32)
    horizontal, free = boxes.group([level])
    assert len(horizontal) == 1 and free == []
    horizontal, free = boxes.group([tilted])
    assert horizontal == [] and len(free) == 1


# --- the networks -------------------------------------------------------------------------------

def test_the_detector_returns_two_channels_at_half_resolution():
    """Region map and affinity map. The half is what ``rescale`` undoes."""
    with torch.no_grad():
        out = CRAFT().eval()(torch.zeros(1, 3, 64, 96))
    assert out.shape == (1, 32, 48, 2)


def test_the_recogniser_collapses_height_and_keeps_width():
    """Height is thrown away on purpose -- a line of text has one row of meaning -- while width
    survives, because width is what CTC steps along."""
    with torch.no_grad():
        out = CRNN(num_class=10).eval()(torch.zeros(2, 1, 64, 128))
    assert out.shape[0] == 2 and out.shape[2] == 10
    assert out.shape[1] > 1


def test_only_the_second_generation_network_is_vendored():
    """Every published variant is second generation, so upstream's ResNet extractor would be a
    class nothing constructs. Its absence is the thing being pinned."""
    import mozo.vendors.easyocr_deploy.crnn as module

    assert not any("ResNet" in name for name in dir(module))


# --- the registry and the adapter ---------------------------------------------------------------

def test_registry_agrees_with_the_adapter():
    """The variant list is written twice -- here and in the adapter -- so that answering "what
    exists" needs no torch import. This is what holds the two copies in step."""
    from mozo.adapters.easyocr import EasyOCRPredictor

    entry = get_model_info(FAMILY)
    assert entry["adapter_class"] == "EasyOCRPredictor"
    assert entry["module"] == "mozo.adapters.easyocr"
    assert entry["task_type"] == "text_recognition"
    assert set(entry["variants"]) == set(EasyOCRPredictor.VARIANTS)


def test_the_adapter_publishes_every_variant_the_vendor_can_build():
    from mozo.adapters.easyocr import EasyOCRPredictor

    assert set(EasyOCRPredictor.VARIANTS) == set(SPECS)


def test_a_variant_that_does_not_exist_is_refused_before_any_download():
    from mozo.adapters.easyocr import EasyOCRPredictor

    with pytest.raises(ValueError, match="Unsupported variant"):
        EasyOCRPredictor("klingon")


# --- with weights ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def english():
    """One variant is enough for the tests that load weights: all five share one network and
    differ only in their alphabet, and the Chinese checkpoint costs 105 MB to prove the same."""
    require_weights(FAMILY, "english")
    from mozo.adapters.easyocr import EasyOCRPredictor

    return EasyOCRPredictor("english", device="cpu")


def test_a_read_carries_text_and_no_class_name(english):
    """PixelFlow keeps the two apart: ``class_name`` is which class out of a vocabulary the model
    was trained on, ``text`` is content it produced that belongs to none."""
    found = english.predict(TEXT_FIXTURES / "printed.png")
    assert [d.text for d in found] == ["Hello World", "OCR 12345", "mozo easyocr"]
    assert all(d.class_name is None and d.class_id is None for d in found)


def test_the_quad_survives_and_the_hull_contains_it(english):
    """Rotated text is the reason the quad is kept; ``bbox`` alone throws the orientation away."""
    found = english.predict(TEXT_FIXTURES / "rotated.png")
    assert len(found) == 2
    for detection in found:
        corners = np.asarray(detection.segments, dtype=float)
        assert corners.shape == (4, 2)
        x1, y1, x2, y2 = detection.bbox
        assert x1 <= corners[:, 0].min() and corners[:, 0].max() <= x2
        assert y1 <= corners[:, 1].min() and corners[:, 1].max() <= y2
    # Genuinely tilted, not an axis-aligned rectangle written as four points.
    first = np.asarray(found[0].segments, dtype=float)
    assert first[0][1] != first[1][1]


def test_an_image_with_no_text_returns_nothing(english):
    assert len(english.predict(TEXT_FIXTURES / "blank.png")) == 0


def test_text_and_segments_survive_to_dict(english):
    """The server returns exactly this, so a field that does not serialise is a field the API
    does not have."""
    payload = english.predict(TEXT_FIXTURES / "sign.png").to_dict()
    entry = payload[0] if isinstance(payload, list) else payload
    assert entry["text"] == "EXIT 42"
    assert len(entry["segments"]) == 4
    assert entry["class_name"] is None


def test_reading_the_same_page_twice_gives_the_same_answer(english):
    """The reader holds no state between calls, and the answer should not depend on that being
    true by accident."""
    first = english.predict(TEXT_FIXTURES / "wide.png")
    second = english.predict(TEXT_FIXTURES / "wide.png")
    assert [d.text for d in first] == [d.text for d in second]
    assert [d.confidence for d in first] == [d.confidence for d in second]


def test_level_lines_come_back_before_tilted_ones(english):
    """Upstream's per-line ordering, which mozo reproduces. It is not a reading order and the
    adapter does not present it as one."""
    found = english.predict(TEXT_FIXTURES / "mixed.png")
    tilted = [i for i, d in enumerate(found)
              if abs(np.asarray(d.segments, float)[0][1] - np.asarray(d.segments, float)[1][1]) > 1]
    level = [i for i, d in enumerate(found) if i not in tilted]
    assert level and tilted
    assert max(level) < min(tilted)
