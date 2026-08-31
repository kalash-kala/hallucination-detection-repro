"""csx_extract — component 1: model + rows + images -> reduced internal states.

GPU side. Runs one pair at a time, resumable from per-shard checkpoints. Knows
nothing about arms, contrasts, cohorts or medians: its only output contract is
store_spec/STORE_CONTRACT.md.

Imports neither csx_probe nor csx_report.
"""
