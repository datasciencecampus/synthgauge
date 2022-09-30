"""A library for evaluating synthetic data."""

from . import datasets, metrics, plot, utils
from .evaluator import Evaluator

__version__ = "2.1.0"

__all__ = ["Evaluator", "datasets", "metrics", "plot", "utils", "__version__"]
