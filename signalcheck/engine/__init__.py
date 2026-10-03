"""Deterministic statistical engine that decides every verdict.

The entry point is :func:`analyse` (``preprocess -> checks -> verdict ->
confidence -> change_my_mind``); :func:`compare_sources` summarises several
sources' verdicts over a common calendar window.
"""

from signalcheck.engine.cross_source import compare_sources
from signalcheck.engine.pipeline import Analysis, analyse, analyse_detailed

__all__ = ["Analysis", "analyse", "analyse_detailed", "compare_sources"]
