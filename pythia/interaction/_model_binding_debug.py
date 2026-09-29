"""Opt-in diagnostics for resolved model bindings; never resume authority."""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile

from .model_catalog import ModelBinding, thaw_json


MAX_DEBUG_MODEL_BINDING_BYTES = 262_144


def debug_model_binding_path(log_path):
    path = Path(log_path)
    return path.with_name(path.name + ".model-binding.json")


def _debug_entry(binding):
    if not isinstance(binding, ModelBinding):
        raise TypeError("debug bindings must be ModelBinding values")
    spec = binding.spec
    return {
        "selector": binding.selector,
        "canonical": None if spec is None else spec.name,
        "source": binding.origin,
        "api_explicit": binding.api_explicit,
        "endpoint": {
            "api": binding.endpoint.api,
            "url": binding.endpoint.url,
            "model": binding.endpoint.model,
            "auth": binding.endpoint.auth,
        },
        "limits": {
            name: getattr(binding.limits, name)
            for name in (
                "auto_compact_context_tokens",
                "max_context_tokens",
                "max_output_tokens",
            )
        },
        "request_params": thaw_json(binding.request_params),
        "responses": (
            None if spec is None or spec.responses is None
            else vars(spec.responses)
        ),
    }


def save_debug_model_bindings(path, bindings):
    """Atomically write diagnostics, returning a safe warning on failure."""
    path = Path(path)
    temporary = None
    try:
        if not isinstance(bindings, Mapping) or any(
            not isinstance(key, str) or not key for key in bindings
        ):
            raise TypeError("debug bindings must be a string-keyed mapping")
        payload = json.dumps(
            {
                "version": 1,
                "bindings": {
                    key: _debug_entry(binding)
                    for key, binding in bindings.items()
                },
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        if len(payload.encode("utf-8")) > MAX_DEBUG_MODEL_BINDING_BYTES:
            raise ValueError("debug model-binding snapshot exceeds size limit")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".model-binding-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except (OSError, TypeError, ValueError, RecursionError):
        return f"Warning: could not save debug model-binding snapshot: {path}"
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return None


__all__ = ["debug_model_binding_path", "save_debug_model_bindings"]
