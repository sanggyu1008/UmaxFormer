"""Model factory for ENSO experiments."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

try:
    from src.models.umaxformer import UmaxFormer
except ImportError:
    UmaxFormer = None  # type: ignore[assignment]


MODEL_REGISTRY: Dict[str, Any] = {
    "umaxformer_v1": UmaxFormer,
    "umaxformer_v2": UmaxFormer,
    "umaxformer_v2_sba1": UmaxFormer,
}


def get_model_class(model_name: str) -> Callable[..., Any]:
    """Return the model class registered for a config `model.name` value."""
    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model.name {model_name!r}. Available: {choices}") from exc
    if model_cls is None:
        raise ImportError(
            "UmaxFormer is not importable. Restore "
            "src/models/umaxformer.py before instantiating this model."
        )
    return model_cls


def build_model(config: Mapping[str, Any]) -> Any:
    """Instantiate the model described by a loaded experiment config."""
    model_config = dict(config.get("model", {}))
    model_name = model_config.pop("name", "umaxformer_v1")
    model_config.pop("implementation", None)
    model_config.pop("promoted_from", None)
    return get_model_class(model_name)(**model_config)
