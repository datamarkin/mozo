"""
Device detection and management for Mozo.

Provides automatic detection of the best available compute device
(CUDA GPU, Apple MPS, or CPU) with caching for performance.
"""

_default_device = None


def get_default_device() -> str:
    """
    Auto-detect the best available compute device.

    Detection priority:
    1. CUDA (NVIDIA GPU) - if torch.cuda.is_available()
    2. MPS (Apple Silicon) - if torch.backends.mps.is_available()
    3. CPU - fallback

    The result is cached after first call for performance.

    Returns:
        str: Device identifier ('cuda', 'mps', or 'cpu')

    Example:
        >>> from mozo.device import get_default_device
        >>> device = get_default_device()
        >>> print(f"Using device: {device}")
    """
    global _default_device

    if _default_device is not None:
        return _default_device

    try:
        import torch

        if torch.cuda.is_available():
            _default_device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _default_device = 'mps'
        else:
            _default_device = 'cpu'
    except ImportError:
        # PyTorch not installed - default to CPU
        _default_device = 'cpu'

    return _default_device


def reset_default_device():
    """
    Reset the cached default device (useful for testing).
    """
    global _default_device
    _default_device = None
