from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wayla")
except PackageNotFoundError:
    # package is not installed
    pass
