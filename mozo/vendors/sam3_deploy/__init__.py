# SPDX-License-Identifier: Apache-2.0
"""Deployment-only SAM 3 image segmentation.

Derived from ``transformers/models/sam3`` (Apache-2.0), not from ``facebookresearch/sam3``, whose
code ships under the SAM License. See ``PROVENANCE.md``.

The model weights are Meta's and carry the SAM License; mozo does not redistribute them.
"""

from .config import SPEC, Spec, TrunkSpec

__all__ = ["SPEC", "Spec", "TrunkSpec"]
