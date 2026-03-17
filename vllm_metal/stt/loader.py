# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text model loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.stt.qwen3_asr.config import Qwen3ASRConfig
from vllm_metal.stt.qwen3_asr.model import Qwen3ASRModel
from vllm_metal.stt.whisper.config import WhisperConfig
from vllm_metal.stt.whisper.model import WhisperModel

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None  # type: ignore[assignment]

# Supported floating-point dtypes for STT model loading.
_SUPPORTED_LOAD_DTYPES = frozenset({mx.float16, mx.float32, mx.bfloat16})


def _read_config(model_path: Path) -> dict:
    """Read and return config.json from a model directory."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path) as f:
        return json.load(f)


def _load_weights(model_path: Path) -> dict[str, mx.array]:
    """Load model weights from safetensors or npz files."""
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("*.npz"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files in {model_path}")

    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(str(wf)))
    return weights


def _resolve_model_path(model_path: str | Path) -> Path:
    """Resolve model path, downloading from HF if needed."""
    model_path = Path(model_path)
    if not model_path.exists():
        if snapshot_download is None:
            raise ValueError(
                f"Could not download model {model_path}: huggingface_hub is not installed"
            )
        try:
            model_path = Path(snapshot_download(repo_id=str(model_path)))
        except OSError as e:
            raise ValueError(f"Could not download model: {model_path}") from e
    return model_path


def _validate_load_dtype(dtype: mx.Dtype) -> None:
    """Validate the floating-point dtype used for model loading."""
    if dtype not in _SUPPORTED_LOAD_DTYPES:
        names = ", ".join(sorted(str(d) for d in _SUPPORTED_LOAD_DTYPES))
        raise TypeError(
            f"Unsupported STT model dtype: {dtype!r}. Must be one of {names}."
        )


def load_model(model_path: str | Path, dtype: mx.Dtype = mx.float16):
    """Load an STT model from a local directory or HuggingFace repo.

    Auto-detects model type from config.json and dispatches to the
    appropriate loader (Whisper or Qwen3-ASR).
    """
    if isinstance(model_path, str) and not model_path.strip():
        raise ValueError(
            "model_path must be a non-empty local path or HuggingFace repo ID."
        )
    _validate_load_dtype(dtype)
    model_path = _resolve_model_path(model_path)
    config_dict = _read_config(model_path)
    model_type = config_dict.get("model_type", "").lower()

    if model_type == "qwen3_asr":
        return _load_qwen3_asr_model(model_path, config_dict, dtype)
    if model_type in ("", "whisper"):
        # Default to Whisper for backward compatibility
        return _load_whisper_model(model_path, config_dict, dtype)
    raise ValueError(
        f"Unsupported STT model_type: {model_type!r}. "
        "Expected 'whisper' or 'qwen3_asr'."
    )


def _load_and_init_model(model, model_path: Path, config_dict: dict):
    """Shared loader: quantize, sanitize, load weights, and eval."""
    weights = _load_weights(model_path)

    quantization = config_dict.get("quantization")
    if quantization is not None:

        def class_predicate(p, m):
            return isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights

        nn.quantize(model, **quantization, class_predicate=class_predicate)

    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def _load_whisper_model(
    model_path: Path, config_dict: dict, dtype: mx.Dtype
) -> WhisperModel:
    """Load a Whisper model from config and weights."""
    config = WhisperConfig.from_dict(config_dict)
    model = WhisperModel(config, dtype)
    return _load_and_init_model(model, model_path, config_dict)


def _load_qwen3_asr_model(model_path: Path, config_dict: dict, dtype: mx.Dtype):
    """Load a Qwen3-ASR model from config and weights."""
    config = Qwen3ASRConfig.from_dict(config_dict)
    model = Qwen3ASRModel(config, dtype)
    return _load_and_init_model(model, model_path, config_dict)
