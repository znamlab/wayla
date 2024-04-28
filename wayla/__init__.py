from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wayla")
except PackageNotFoundError:
    # package is not installed
    pass

from . import eye_model_fitting
from . import diagnostics
from .eye_tracking import run_all
