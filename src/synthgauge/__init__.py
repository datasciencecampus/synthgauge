# flake8: noqa
import pkg_resources

from . import datasets, evaluate, metrics, plot, utils
from .evaluate import Evaluator

try:
    __version__ = pkg_resources.get_distribution("synthgauge").version
except pkg_resources.DistributionNotFound:
    # Raised when package has not been installed e.g. just
    # src added to path.
    __version__ = None
