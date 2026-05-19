# As there is a file called __init__.py, this whole part becomes a package

"""Transparent TWFE and Sun-Abraham event-study estimators."""

from twfeiw.api import EventStudyResult, TWFEResult, event_study, twfe

__version__ = "0.1.0"

# these are for public naming
__all__ = ["EventStudyResult", "TWFEResult", "__version__", "event_study", "twfe"]
