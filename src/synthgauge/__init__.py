# flake8: noqa
import pkg_resources

from . import datasets
from .evaluate import Evaluator
from . import evaluate
from . import metrics
from . import plot
from . import utils

try:
    __version__ = pkg_resources.get_distribution('synthgauge').version
except pkg_resources.DistributionNotFound:
    # Raised when package has not been installed e.g. just
    # src added to path.
    __version__ = None
