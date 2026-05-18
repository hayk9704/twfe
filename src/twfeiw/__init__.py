# As there is a file called __init__.py, this whole part becomes a package

"""Transparent TWFE and Sun-Abraham event-study estimators."""

from twfeiw.api import TWFEResult, twfe

__version__ = "0.1.0"

__all__ = ["TWFEResult", "__version__", "twfe"]
