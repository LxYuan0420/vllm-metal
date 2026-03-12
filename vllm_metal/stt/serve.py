# SPDX-License-Identifier: Apache-2.0
"""Serve-boundary helpers for Speech-to-Text requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np


@dataclass(frozen=True)
class STTRequestInput:
    """Normalized STT request data consumed by the runtime path."""

    req_id: str
    prompt_token_ids: tuple[int, ...]
    input_features: Any


class VLLMSTTRequestAdapter:
    """Boundary adapter that normalizes raw vLLM STT requests."""

    @classmethod
    def from_vllm_request(cls, request: Any) -> STTRequestInput:
        """Normalize a vLLM request object for the STT runtime path."""
        req_id = cls._get_request_id(request)
        prompt_token_ids = cls._get_request_prompt_token_ids(request, req_id)
        mm_features = cls._get_request_mm_features(request, req_id)
        normalized_prompt_token_ids = cls._normalize_prompt_token_ids(
            req_id, prompt_token_ids
        )

        return STTRequestInput(
            req_id=req_id,
            prompt_token_ids=normalized_prompt_token_ids,
            input_features=cls._extract_input_features(req_id, mm_features),
        )

    @staticmethod
    def _get_request_id(request: Any) -> str:
        """Return the request id from a vLLM STT request object."""
        try:
            return str(request.req_id)
        except AttributeError as exc:
            raise ValueError("STT request is missing req_id.") from exc

    @staticmethod
    def _get_request_prompt_token_ids(request: Any, req_id: str) -> Any:
        """Return prompt token ids from a vLLM STT request object."""
        try:
            return request.prompt_token_ids
        except AttributeError as exc:
            raise ValueError(
                f"STT request {req_id!r} is missing prompt_token_ids."
            ) from exc

    @staticmethod
    def _get_request_mm_features(request: Any, req_id: str) -> Any:
        """Return multimodal features from a vLLM STT request object."""
        try:
            return request.mm_features
        except AttributeError as exc:
            raise ValueError(f"STT request {req_id!r} is missing mm_features.") from exc

    @staticmethod
    def _normalize_prompt_token_ids(
        req_id: str, prompt_token_ids: Any
    ) -> tuple[int, ...]:
        """Normalize request prompt token ids for the STT runtime path."""
        if prompt_token_ids is None:
            return ()

        if isinstance(prompt_token_ids, (str, bytes)):
            raise ValueError(f"STT request {req_id!r} has invalid prompt_token_ids.")

        try:
            tokens = tuple(prompt_token_ids)
        except TypeError as exc:
            raise ValueError(
                f"STT request {req_id!r} has invalid prompt_token_ids."
            ) from exc

        if not all(
            isinstance(tok, Integral) and not isinstance(tok, bool) for tok in tokens
        ):
            raise ValueError(f"STT request {req_id!r} has invalid prompt_token_ids.")

        return tuple(int(tok) for tok in tokens)

    @classmethod
    def _extract_input_features(cls, req_id: str, mm_features: Any) -> Any:
        """Extract raw input features from a vLLM multimodal payload."""
        if not isinstance(mm_features, list) or not mm_features:
            raise ValueError(
                f"STT request {req_id!r} must include non-empty mm_features."
            )

        payload = cls._resolve_feature_payload(req_id, mm_features[0])
        input_features = payload.get("input_features")

        if input_features is None:
            raise ValueError(f"STT request {req_id!r} must include input_features.")

        if isinstance(input_features, np.ndarray):
            return input_features

        try:
            unwrapped_input_features = input_features.data
        except AttributeError:
            return input_features

        if unwrapped_input_features is None:
            raise ValueError(f"STT request {req_id!r} must include input_features.")

        return unwrapped_input_features

    @staticmethod
    def _resolve_feature_payload(req_id: str, feature: Any) -> Mapping[str, Any]:
        """Return the mapping payload for one multimodal feature entry."""
        if isinstance(feature, Mapping):
            return feature

        try:
            payload = feature.data
        except AttributeError as exc:
            raise ValueError(
                f"STT request {req_id!r} must include input_features."
            ) from exc

        if isinstance(payload, Mapping):
            return payload

        raise ValueError(f"STT request {req_id!r} must include input_features.")
