"""Backend registry."""

from __future__ import annotations

from typing import Callable

from private_markets_cd.backends.base import CashFlowBackend
from private_markets_cd.backends.mrsr import MeanRevertingBackend


def _lstm_factory(**kwargs) -> CashFlowBackend:
    from private_markets_cd.backends.lstm import LSTMBackend

    return LSTMBackend(**kwargs)


_REGISTRY: dict[str, Callable[..., CashFlowBackend]] = {
    "mrsr": lambda **kw: MeanRevertingBackend(**kw),
    "mean_reverting": lambda **kw: MeanRevertingBackend(**kw),
    "lstm": _lstm_factory,
}


def list_backends() -> list[str]:
    return sorted(set(_REGISTRY) - {"mean_reverting"}) + ["mean_reverting"]


def get_backend(name: str = "mrsr", **kwargs) -> CashFlowBackend:
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown backend {name!r}. Available: {list_backends()}")
    return _REGISTRY[key](**kwargs)
