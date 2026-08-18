# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Behaviour in this module is reimplemented from HuggingFace Transformers
# (https://github.com/huggingface/transformers), Apache License 2.0.
# Copyright 2023 The HuggingFace Inc. team. All Rights Reserved.
# ------------------------------------------------------------------------
"""Stand-ins for the HuggingFace Transformers base classes used by the vendored DINOv2 backbone.

``dinov2_with_windowed_attn`` defines every leaf module itself and only reaches into Transformers for a configuration
container, an ``nn.Module`` base with a weight initializer, a backbone mixin, an activation lookup, and two output
containers.  Reimplementing that surface here removes the ``transformers`` dependency — and with it
``huggingface-hub``, ``tokenizers``, ``safetensors``, and ``regex`` — from the deployment install.

The reimplementation is deliberately behaviour-preserving rather than minimal:

* ``_attn_implementation`` defaults to ``"sdpa"``, matching what ``PretrainedConfig`` resolves to for this backbone
  under ``transformers`` 5.x (verified against 5.8.0). Attention implementation changes numerics, so this default is
  load-bearing for output parity.
* ``ACT2FN["gelu"]`` maps to :class:`torch.nn.GELU`, which is bitwise identical to the ``GELUActivation`` module
  Transformers returns. Both are parameterless ``nn.Module`` instances, so the state-dict layout is unchanged.
* :meth:`PreTrainedModel.post_init` still applies ``_init_weights`` to the module tree. Deployment always overwrites
  those values with checkpoint weights, but keeping the call means a partially-loaded model degrades the same way it
  does upstream instead of silently keeping PyTorch's default initialization.

Not carried over: ``from_pretrained`` / ``save_pretrained`` (checkpoints are loaded by
:mod:`rfdetr_deploy.weights`), head pruning, device-map sharding, and the doc-string decorators.
"""

from __future__ import annotations

__all__ = [
    "ACT2FN",
    "BackboneMixin",
    "BackboneOutput",
    "BaseModelOutput",
    "PretrainedConfig",
    "PreTrainedModel",
    "torch_int",
]

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

# Transformers exposes a large activation registry; this backbone's configs only ever request "gelu".
# The remaining entries are the ones a DINOv2-family config could plausibly carry, mapped to the same
# implementations Transformers uses.
ACT2FN: dict[str, nn.Module] = {
    "gelu": nn.GELU(),
    "gelu_new": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "swish": nn.SiLU(),
}


def torch_int(x: Any) -> Any:
    """Cast *x* to an int, preserving tensor-ness under ``torch.jit`` tracing.

    Reimplemented from ``transformers.utils.torch_int``. Under tracing, converting a tensor to a Python ``int`` would
    bake the value into the graph as a constant; returning an int64 tensor keeps it dynamic.

    Args:
        x: A Python number or a ``torch.Tensor`` holding a single value.

    Returns:
        ``x`` as an int64 ``Tensor`` when tracing a ``Tensor``, otherwise as a Python ``int``.

    Examples:
        >>> torch_int(3.0)
        3
        >>> torch_int(torch.tensor(5.0))
        5
    """
    return x.to(torch.int64) if torch.jit.is_tracing() and isinstance(x, Tensor) else int(x)


@dataclass
class BaseModelOutput:
    """Container returned by the encoder when ``return_dict=True``.

    Reimplemented from ``transformers.modeling_outputs.BaseModelOutput``. RF-DETR runs the backbone with
    ``return_dict=False``, so this exists for API compatibility rather than for the deployment path.

    Attributes:
        last_hidden_state: Final-layer hidden states.
        hidden_states: Per-layer hidden states when requested.
        attentions: Per-layer attention weights when requested.

    Examples:
        >>> out = BaseModelOutput(last_hidden_state=torch.zeros(1, 2, 3))
        >>> tuple(out.last_hidden_state.shape)
        (1, 2, 3)
    """

    last_hidden_state: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class BackboneOutput:
    """Container returned by the backbone when ``return_dict=True``.

    Reimplemented from ``transformers.modeling_outputs.BackboneOutput``. RF-DETR runs the backbone with
    ``return_dict=False``, so this exists for API compatibility rather than for the deployment path.

    Attributes:
        feature_maps: Selected stage feature maps.
        hidden_states: Per-layer hidden states when requested.
        attentions: Per-layer attention weights when requested.

    Examples:
        >>> out = BackboneOutput(feature_maps=(torch.zeros(1, 4, 2, 2),))
        >>> len(out.feature_maps)
        1
    """

    feature_maps: tuple[Tensor, ...] = field(default_factory=tuple)
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


class PretrainedConfig:
    """Attribute-bag configuration base, reimplemented from ``transformers.configuration_utils.PretrainedConfig``.

    Only the behaviour the vendored backbone relies on is kept: unknown keyword arguments become attributes, the
    output-control flags carry Transformers' defaults, and ``out_features`` / ``out_indices`` are exposed as properties
    over the ``_out_features`` / ``_out_indices`` pair that subclasses populate.

    Args:
        **kwargs: Arbitrary configuration values. Recognised keys (``return_dict``, ``output_hidden_states``,
            ``output_attentions``, ``attn_implementation``, ``torchscript``) get Transformers' defaults when absent;
            every other key is set verbatim as an attribute.

    Examples:
        >>> config = PretrainedConfig(hidden_size=8, return_dict=False)
        >>> config.hidden_size, config.return_dict, config._attn_implementation
        (8, False, 'sdpa')
    """

    def __init__(self, **kwargs: Any) -> None:
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.torchscript = kwargs.pop("torchscript", False)
        # Transformers resolves this to "sdpa" for backbones declaring `_supports_sdpa`, which every
        # released RF-DETR checkpoint was exported and evaluated under.
        self._attn_implementation = kwargs.pop("attn_implementation", None) or "sdpa"
        self.stage_names: list[str] = []
        self._out_features: list[str] | None = None
        self._out_indices: list[int] | None = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def out_features(self) -> list[str] | None:
        """Names of the backbone stages whose feature maps are returned."""
        return self._out_features

    @property
    def out_indices(self) -> list[int] | None:
        """Indices of the backbone stages whose feature maps are returned."""
        return self._out_indices

    def verify_out_features_out_indices(self) -> None:
        """Validate ``_out_features`` / ``_out_indices`` against ``stage_names``.

        Reimplemented from ``transformers.utils.backbone_utils.verify_out_features_out_indices``.

        Raises:
            ValueError: If ``stage_names`` is unset, if a requested feature name is not a known stage, if an index is
                out of range, or if both fields are set to different lengths.

        Examples:
            >>> config = PretrainedConfig()
            >>> config.stage_names = ["stem", "stage1", "stage2"]
            >>> config._out_features, config._out_indices = ["stage2"], [2]
            >>> config.verify_out_features_out_indices()
        """
        if not self.stage_names:
            raise ValueError("stage_names must be set for the backbone to verify out_features / out_indices.")
        if self._out_features is not None:
            unknown = [name for name in self._out_features if name not in self.stage_names]
            if unknown:
                raise ValueError(f"out_features must be a subset of stage_names {self.stage_names}, got {unknown}")
        if self._out_indices is not None:
            n = len(self.stage_names)
            invalid = [idx for idx in self._out_indices if not (-n <= idx < n)]
            if invalid:
                raise ValueError(f"out_indices must be within the stage range [-{n}, {n}), got {invalid}")
        if (
            self._out_features is not None
            and self._out_indices is not None
            and len(self._out_features) != len(self._out_indices)
        ):
            raise ValueError("out_features and out_indices should have the same length if both are set")


class PreTrainedModel(nn.Module):
    """``nn.Module`` base carrying a config, reimplemented from ``transformers.modeling_utils.PreTrainedModel``.

    Subclasses keep their Transformers-style class attributes (``config_class``, ``base_model_prefix``, and so on) so
    the vendored backbone file reads the same as upstream; nothing here consumes them, they are retained as
    documentation of the original contract.

    Args:
        config: The model configuration.

    Examples:
        >>> class _Tiny(PreTrainedModel):
        ...     def __init__(self, config):
        ...         super().__init__(config)
        ...         self.linear = nn.Linear(2, 2)
        >>> model = _Tiny(PretrainedConfig())
        >>> model.config.return_dict
        True
    """

    config_class: type[PretrainedConfig] | None = None
    base_model_prefix: str = ""
    main_input_name: str = "pixel_values"
    supports_gradient_checkpointing: bool = False
    _no_split_modules: list[str] = []
    _supports_sdpa: bool = False

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize *module*'s parameters. Subclasses override; the base is a no-op."""

    def post_init(self) -> None:
        """Apply ``_init_weights`` across the module tree, mirroring Transformers' ``post_init``."""
        self.apply(self._init_weights)

    def _gradient_checkpointing_func(self, *args: Any, **kwargs: Any) -> Any:
        """Reject gradient checkpointing, which this deployment-only build does not carry.

        Raises:
            RuntimeError: Always. The call sites are guarded by ``self.training``, so reaching this means the model
                was put in train mode, which is outside what this package supports.
        """
        raise RuntimeError(
            "Gradient checkpointing is not available in the deployment-only build; this model is inference-only."
        )


class BackboneMixin:
    """Backbone stage bookkeeping, reimplemented from ``transformers.utils.backbone_utils.BackboneMixin``.

    Only the ``_init_transformers_backbone`` path is kept — the timm and neural-architecture-search branches of the
    upstream mixin are unreachable for this backbone.

    Examples:
        >>> class _Tiny(BackboneMixin):
        ...     def __init__(self, config):
        ...         self.config = config
        >>> config = PretrainedConfig()
        >>> config.stage_names = ["stem", "stage1"]
        >>> config._out_features, config._out_indices = ["stage1"], [1]
        >>> backbone = _Tiny(config)
        >>> backbone._init_transformers_backbone()
        >>> backbone.out_features
        ['stage1']
    """

    config: PretrainedConfig

    def _init_transformers_backbone(self) -> None:
        """Copy stage names off the config and validate the requested outputs."""
        self.stage_names = self.config.stage_names
        self.config.verify_out_features_out_indices()
        self.num_features: list[int] | None = None

    @property
    def out_features(self) -> list[str] | None:
        """Names of the stages whose feature maps this backbone returns."""
        return self.config._out_features

    @property
    def out_indices(self) -> list[int] | None:
        """Indices of the stages whose feature maps this backbone returns."""
        return self.config._out_indices
