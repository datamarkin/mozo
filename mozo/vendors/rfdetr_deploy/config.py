# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Architecture specifications for the released RF-DETR model variants.

Upstream RF-DETR expresses these as Pydantic ``ModelConfig`` subclasses carrying both architecture and training
hyperparameters.  Deployment needs only the architecture half and no validation framework, so each variant is a frozen
dataclass instead — which is also what removes ``pydantic`` from the install.

Every field here was dumped from the upstream resolved builder namespace rather than transcribed by hand, so the values
match what ``rfdetr`` feeds to ``build_model()`` for the same variant.  Across all nine released variants only 13 fields
actually differ; the rest are shared and carry their upstream value as the dataclass default.

The XLarge and 2XLarge detection models are deliberately absent: they ship in the separate ``rfdetr_plus`` package
under the PML 1.0 license, not Apache 2.0.
"""

from __future__ import annotations

__all__ = ["MODEL_SPECS", "ModelSpec", "get_spec"]

from dataclasses import dataclass, field, replace
from typing import Literal


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build one RF-DETR variant and post-process its outputs.

    Field names mirror the upstream builder namespace so this dataclass can be passed straight to
    :func:`rfdetr_deploy.models.build_model`, which reads its inputs by attribute.

    Attributes:
        name: Variant identifier, e.g. ``"rfdetr-small"``.
        weights_file: Canonical checkpoint filename for the COCO-pretrained release.
        resolution: Square input side length in pixels the checkpoint was trained at.
        patch_size: ViT patch size. Input dimensions must be divisible by ``patch_size * num_windows``.
        num_windows: Windowed-attention window count along each axis.
        positional_encoding_size: Backbone positional-embedding grid side, ``resolution // patch_size``.
        dec_layers: Decoder layer count.
        num_queries: Object queries evaluated at inference (one group).
        num_select: Detections kept by the top-k selection in post-processing.
        segmentation_head: Whether the variant carries a mask head.
        use_grouppose_keypoints: Whether the variant carries a keypoint head.
        num_keypoints_per_class: Keypoint count per class slot; empty for non-keypoint variants.
        dual_projector: Whether the backbone runs a second projector for keypoint cross-attention.
        dual_projector_kp_only: Whether the second projector feeds only the keypoint path.
        num_classes: Class-slot count excluding background; the head has ``num_classes + 1`` outputs.
        encoder: Backbone identifier.
        hidden_dim: Decoder width.
        sa_nheads: Decoder self-attention heads.
        ca_nheads: Decoder cross-attention heads.
        dec_n_points: Deformable-attention sampling points per head per level.
        group_detr: Query groups packed into the checkpoint; only group 0 is read at inference.
        out_feature_indexes: Backbone stages forwarded to the decoder.
        projector_scale: Feature-pyramid levels fed to cross-attention.
        dim_feedforward: Decoder FFN width.
        mask_downsample_ratio: Mask-head output stride relative to the input.
        postprocess_trace_alpha: Keypoint uncertainty fusion weight applied to detection scores.
        num_channels: Input channel count.
        vit_encoder_num_layers: Backbone depth.
        position_embedding: Positional-encoding type for the decoder.
        decoder_norm: Decoder normalization type.
        two_stage: Whether the decoder uses two-stage query selection.
        lite_refpoint_refine: Whether reference-point refinement uses the lite path.
        bbox_reparam: Whether boxes are reparameterized.
        layer_norm: Whether the projector applies layer normalization.
        use_cls_token: Whether the backbone prepends a CLS token.
        num_decoder_registers: Decoder register-token count.
        grouppose_keypoint_dim_downscale: Keypoint branch width divisor.
        keypoint_cross_attn: Whether the keypoint branch runs its own cross-attention.
        inter_instance_kp_attn: Whether keypoints attend across instances.

    Examples:
        >>> spec = get_spec("rfdetr-small")
        >>> spec.resolution, spec.dec_layers, spec.segmentation_head
        (512, 3, False)
    """

    name: str
    weights_file: str
    resolution: int
    patch_size: int
    num_windows: int
    positional_encoding_size: int
    dec_layers: int
    num_queries: int
    num_select: int

    segmentation_head: bool = False
    use_grouppose_keypoints: bool = False
    num_keypoints_per_class: list[int] = field(default_factory=list)
    dual_projector: bool = False
    dual_projector_kp_only: bool = False

    num_classes: int = 90
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    group_detr: int = 13
    out_feature_indexes: list[int] = field(default_factory=lambda: [3, 6, 9, 12])
    projector_scale: list[Literal["P3", "P4", "P5"]] = field(default_factory=lambda: ["P4"])
    dim_feedforward: int = 2048
    mask_downsample_ratio: int = 4
    postprocess_trace_alpha: float = 0.2
    num_channels: int = 3

    vit_encoder_num_layers: int = 12
    position_embedding: str = "sine"
    decoder_norm: str = "LN"
    two_stage: bool = True
    lite_refpoint_refine: bool = True
    bbox_reparam: bool = True
    layer_norm: bool = True
    use_cls_token: bool = False
    num_decoder_registers: int = 0
    grouppose_keypoint_dim_downscale: int = 1
    keypoint_cross_attn: bool = True
    inter_instance_kp_attn: bool = False

    # Builder inputs with no deployment meaning, kept so `build_model()` reads the same attributes it does upstream.
    pretrained_encoder: None = None
    window_block_indexes: None = None
    drop_path: float = 0.0
    rms_norm: bool = False
    freeze_encoder: bool = False
    backbone_lora: bool = False
    force_no_pretrain: bool = False
    gradient_checkpointing: bool = False
    aux_loss: bool = True
    dropout: float = 0.0
    encoder_only: bool = False
    backbone_only: bool = False
    device: str = "cpu"
    pretrain_weights: str | None = None
    num_feature_levels: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        """Normalize sequence fields to lists and derive ``num_feature_levels``.

        The vendored builders compare ``projector_scale`` against ``sorted(projector_scale)`` and slice
        ``num_keypoints_per_class``; a tuple would make the former compare unequal and fail an assertion, so
        sequences are coerced here rather than trusted from the caller.
        """
        object.__setattr__(self, "out_feature_indexes", list(self.out_feature_indexes))
        object.__setattr__(self, "projector_scale", list(self.projector_scale))
        object.__setattr__(self, "num_keypoints_per_class", list(self.num_keypoints_per_class))
        object.__setattr__(self, "num_feature_levels", len(self.projector_scale))

    def with_overrides(self, **overrides: object) -> ModelSpec:
        """Return a copy of this spec with *overrides* applied.

        Args:
            **overrides: Field values to replace.

        Returns:
            A new frozen ``ModelSpec``.

        Examples:
            >>> get_spec("rfdetr-nano").with_overrides(num_classes=3).num_classes
            3
        """
        return replace(self, **overrides)  # type: ignore[arg-type]


_SPEC_LIST: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="rfdetr-nano",
        weights_file="rf-detr-nano.pth",
        resolution=384,
        patch_size=16,
        num_windows=2,
        positional_encoding_size=24,
        dec_layers=2,
        num_queries=300,
        num_select=300,
    ),
    ModelSpec(
        name="rfdetr-small",
        weights_file="rf-detr-small.pth",
        resolution=512,
        patch_size=16,
        num_windows=2,
        positional_encoding_size=32,
        dec_layers=3,
        num_queries=300,
        num_select=300,
    ),
    ModelSpec(
        name="rfdetr-medium",
        weights_file="rf-detr-medium.pth",
        resolution=576,
        patch_size=16,
        num_windows=2,
        positional_encoding_size=36,
        dec_layers=4,
        num_queries=300,
        num_select=300,
    ),
    ModelSpec(
        name="rfdetr-large",
        weights_file="rf-detr-large-2026.pth",
        resolution=704,
        patch_size=16,
        num_windows=2,
        positional_encoding_size=44,
        dec_layers=4,
        num_queries=300,
        num_select=300,
    ),
    ModelSpec(
        name="rfdetr-seg-nano",
        weights_file="rf-detr-seg-nano.pt",
        resolution=312,
        patch_size=12,
        num_windows=1,
        positional_encoding_size=26,
        dec_layers=4,
        num_queries=100,
        num_select=100,
        segmentation_head=True,
    ),
    ModelSpec(
        name="rfdetr-seg-small",
        weights_file="rf-detr-seg-small.pt",
        resolution=384,
        patch_size=12,
        num_windows=2,
        positional_encoding_size=32,
        dec_layers=4,
        num_queries=100,
        num_select=100,
        segmentation_head=True,
    ),
    ModelSpec(
        name="rfdetr-seg-medium",
        weights_file="rf-detr-seg-medium.pt",
        resolution=432,
        patch_size=12,
        num_windows=2,
        positional_encoding_size=36,
        dec_layers=5,
        num_queries=200,
        num_select=200,
        segmentation_head=True,
    ),
    ModelSpec(
        name="rfdetr-seg-large",
        weights_file="rf-detr-seg-large.pt",
        resolution=504,
        patch_size=12,
        num_windows=2,
        positional_encoding_size=42,
        dec_layers=5,
        num_queries=200,
        num_select=200,
        segmentation_head=True,
    ),
    ModelSpec(
        name="rfdetr-keypoint-preview",
        weights_file="rf-detr-keypoint-preview-xlarge.pth",
        resolution=576,
        patch_size=12,
        num_windows=2,
        positional_encoding_size=48,
        dec_layers=4,
        num_queries=100,
        num_select=100,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
        dual_projector=True,
        dual_projector_kp_only=True,
    ),
)

MODEL_SPECS: dict[str, ModelSpec] = {spec.name: spec for spec in _SPEC_LIST}


def get_spec(name: str) -> ModelSpec:
    """Look up a variant spec by name.

    Args:
        name: Variant identifier, e.g. ``"rfdetr-seg-nano"``.

    Returns:
        The frozen :class:`ModelSpec` for *name*.

    Raises:
        KeyError: If *name* is not a known variant.

    Examples:
        >>> get_spec("rfdetr-seg-nano").patch_size
        12
        >>> get_spec("rfdetr-xlarge")
        Traceback (most recent call last):
        ...
        KeyError: "unknown model 'rfdetr-xlarge'; available: ..."
    """
    try:
        return MODEL_SPECS[name]
    except KeyError:
        available = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"unknown model {name!r}; available: {available}") from None
