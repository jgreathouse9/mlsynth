"""Helpers for geo-experiment market selection.

The market-selection workflow nominates candidate test-market sets and scores
them. ``similarity`` holds the treatment-agnostic, estimator-independent
leaves: the unit-by-unit correlation matrix and the ranked-neighbor table that
a candidate generator consumes.
"""
