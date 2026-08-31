"""csx_probe — component 2: store + arms -> atomic per-pair values.

Emits one row per atomic measurement, keyed by pair. It never takes a median and
never sees a cohort: aggregation is csx_report's job, which is what lets any
grouping be chosen after the fact without refitting anything.

Has no torch dependency; tests/test_isolation.py enforces that.
"""
