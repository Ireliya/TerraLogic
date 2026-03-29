"""
Results module for spatialreason.plan

This module contains classes for formatting and storing execution results.

Classes:
    ResultFormatter: Formats results for evaluation and output
    ResultStorage: Stores and tracks execution results
"""

from .result_formatter import ResultFormatter
from .result_storage import ResultStorage

__all__ = ["ResultFormatter", "ResultStorage"]

