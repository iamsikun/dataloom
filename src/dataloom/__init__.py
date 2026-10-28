"""Dataloom: Simulation studies for Generalized Neyman Allocation."""

__version__ = "0.1.0"


# Eagerly import all subpackages that register estimators so that
#     from dataloom.estimators import REGISTRY
# returns a fully-populated registry no matter the entry point.
# Order matters: estimators package first (it owns the registry); then
# adaptive (which uses register), then inference (which uses adaptive).
from . import estimators as _estimators  # noqa: F401, E402
from . import adaptive as _adaptive      # noqa: F401, E402
from . import inference as _inference    # noqa: F401, E402
