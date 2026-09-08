"""Outlier definition (spec section 4): baselines, screening, tiers, gates, recompute.

Pure functions over rows and config. No database access in this package except `recompute`,
which reads raw daily rows and writes derived stats.
"""
