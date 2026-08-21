"""The recogniser -- a CRNN read with CTC.

One rectified, greyscale line of text in; one distribution over the alphabet per horizontal
step out. A VGG stack collapses the crop's height to a single row while leaving its width
alone, two bidirectional LSTMs give every step the context of the whole line, and a linear
layer scores the alphabet at each. Nothing here decides what the line *says* -- collapsing a
step sequence into characters is CTC's job, in :mod:`.text`.

Only the second-generation network is here. EasyOCR published two: this one, and an earlier
ResNet-based variant with 512-wide LSTMs. Every variant mozo publishes is second generation, so
the ResNet is not extracted -- see PROVENANCE.md.
"""

from __future__ import annotations

__all__ = ["CRNN"]

import torch
import torch.nn as nn


class VGGFeatures(nn.Module):
    """Greyscale crop to a ``(B, C, 1, W/4)`` feature strip.

    The pooling is deliberately asymmetric. The first two max pools halve both axes; the last
    two halve only the height. Height is being thrown away on purpose -- a line of text has one
    row of meaning -- while width has to survive, because width is what CTC steps along, and a
    step that spans two characters cannot be split back apart later.
    """

    def __init__(self, input_channel: int = 1, output_channel: int = 256) -> None:
        super().__init__()
        c = [output_channel // 8, output_channel // 4, output_channel // 2, output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, c[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c[0], c[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c[1], c[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(c[2], c[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(c[2], c[3], 3, 1, 1, bias=False), nn.BatchNorm2d(c[3]), nn.ReLU(True),
            nn.Conv2d(c[3], c[3], 3, 1, 1, bias=False), nn.BatchNorm2d(c[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(c[3], c[3], 2, 1, 0), nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ConvNet(x)


class BidirectionalLSTM(nn.Module):
    """One BiLSTM whose two directions are projected back down to ``output_size``."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upstream calls ``rnn.flatten_parameters()`` here, inside a bare try/except, so that
        # DataParallel does not warn on every step. It relays out the weights and changes no
        # number; there is no DataParallel in a deployment package, so it is gone.
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)


class CRNN(nn.Module):
    """Feature extraction, sequence modelling, prediction -- upstream's three stages.

    ``num_class`` is the alphabet plus one: CTC needs a blank symbol, and it takes index 0. See
    :class:`~mozo.vendors.easyocr_deploy.text.Alphabet`.
    """

    def __init__(self, num_class: int, input_channel: int = 1, output_channel: int = 256,
                 hidden_size: int = 256) -> None:
        super().__init__()
        self.FeatureExtraction = VGGFeatures(input_channel, output_channel)
        # The extractor already leaves height at 1, so this pool is a no-op on a well-formed
        # crop. It is kept because it is what the published graph does, and because it is the
        # only thing standing between a mis-sized input and a shape error deep in the LSTM.
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.Prediction = nn.Linear(hidden_size, num_class)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``(B, 1, 64, W)`` greyscale crops to ``(B, T, num_class)`` logits.

        Upstream's signature takes a second ``text`` argument and ignores it. It is there so an
        attention-based head could share the call; this is a CTC model, and no attention head is
        extracted, so the argument is dropped rather than accepted and discarded.
        """
        visual = self.FeatureExtraction(image)
        # ``(B, C, H, W)`` -> ``(B, W, C, H)``, so the pool reduces height and the sequence axis
        # ends up second, which is what ``batch_first`` LSTMs want.
        visual = self.AdaptiveAvgPool(visual.permute(0, 3, 1, 2)).squeeze(3)
        return self.Prediction(self.SequenceModeling(visual).contiguous())
