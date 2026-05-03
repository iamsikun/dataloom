"""Tabular generators with a common fit/sample API."""

from .base import TabularGenerator
from .bootstrap_smoothed import BootstrapSmoothed
from .gaussian_copula import GaussianCopulaGenerator

GENERATORS: dict[str, type[TabularGenerator]] = {
    "bootstrap_smoothed": BootstrapSmoothed,
    "gaussian_copula": GaussianCopulaGenerator,
}


def get_generator(name: str, **kwargs) -> TabularGenerator:
    cls = GENERATORS[name]
    return cls(**kwargs)
