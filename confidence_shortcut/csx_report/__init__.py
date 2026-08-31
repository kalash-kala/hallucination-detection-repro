"""csx_report — component 3: atomic per-pair values -> tables and reports.

The only component that aggregates. Cohorts are a runtime argument, so any
grouping of pairs can be medianed without refitting: LLM vs VLM, per-dataset,
per-model-size, leave-one-out. Runs in seconds.

Imports neither csx_extract nor csx_probe.
"""
