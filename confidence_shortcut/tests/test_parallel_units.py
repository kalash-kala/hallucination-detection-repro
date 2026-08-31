"""Parallel unit execution: same results, same accounting, no oversubscription."""

from __future__ import annotations

import os

import pytest

from csx_common import cli


def _units(n: int, tmp_path):
    """n runnable units -- outputs that do not exist, no unmet preconditions."""
    return [cli.Unit(key=f"p{i}", outputs=(tmp_path / f"p{i}.parquet",))
            for i in range(n)]


def _ok(u: cli.Unit) -> str:
    return f"{u.key} done"


def test_parallel_and_sequential_agree_on_counts(tmp_path, capsys):
    """Widening the pool must not change what ran, only how fast."""
    seq = cli.run_units("t", _units(4, tmp_path), _ok, pair_jobs=1)
    a = capsys.readouterr().out
    par = cli.run_units("t", _units(4, tmp_path), _ok, pair_jobs=4)
    b = capsys.readouterr().out
    assert seq == par == 0
    assert "4 ok, 0 failed, 0 skipped" in a
    assert "4 ok, 0 failed, 0 skipped" in b


def test_every_unit_runs_exactly_once_in_parallel(tmp_path, capsys):
    cli.run_units("t", _units(6, tmp_path), _ok, pair_jobs=3)
    out = capsys.readouterr().out
    for i in range(6):
        assert out.count(f"p{i} done") == 1


def test_one_failure_does_not_cancel_the_pool(tmp_path, capsys):
    """A raise in one unit must cost only that unit, not its in-flight peers."""
    def fn(u: cli.Unit) -> str:
        if u.key == "p2":
            raise ValueError("boom")
        return "fine"

    rc = cli.run_units("t", _units(5, tmp_path), fn, pair_jobs=5)
    out = capsys.readouterr().out
    assert rc == 1
    assert "4 ok, 1 failed, 0 skipped" in out
    assert "[FAIL]" in out and "ValueError: boom" in out


def test_cached_and_blocked_units_are_not_dispatched(tmp_path, capsys):
    done = tmp_path / "cached.parquet"
    done.write_text("x")
    units = [
        cli.Unit(key="cached", outputs=(done,)),
        cli.Unit(key="blocked", outputs=(tmp_path / "b.parquet",),
                 unmet=("needs stage 03",)),
        cli.Unit(key="run", outputs=(tmp_path / "r.parquet",)),
    ]
    seen: list[str] = []

    def fn(u: cli.Unit) -> str:
        seen.append(u.key)
        return ""

    # pair_jobs=1 so the closure's mutation is observable; the dispatch decision
    # under test happens before the pool either way.
    cli.run_units("t", units, fn, pair_jobs=1)
    out = capsys.readouterr().out
    assert seen == ["run"]
    assert "[cached] cached" in out and "[skip] blocked" in out
    assert "1 ok, 0 failed, 2 skipped" in out


@pytest.mark.parametrize("n_units,spec,expected_max", [
    (1, "auto", 1),      # never fan out for a single unit
    (9, "1", 1),         # explicit 1 stays sequential
    (3, "8", 3),         # never more workers than units
])
def test_resolve_pair_jobs_bounds(n_units, spec, expected_max):
    assert cli.resolve_pair_jobs(spec, n_units) == expected_max


def test_auto_scales_with_cores_but_never_exceeds_units():
    n = cli.resolve_pair_jobs("auto", 1000)
    assert 1 <= n <= (os.cpu_count() or 4)


def test_inner_threads_never_oversubscribe():
    """pair_jobs x inner_threads must stay within the core count.

    This is the whole point of dividing the inner pool: 5 processes each taking
    32 OpenBLAS threads on a 32-core box thrashes and can run slower than the
    sequential loop it replaced.
    """
    cores = os.cpu_count() or 4
    for jobs in range(1, cores + 1):
        assert jobs * cli._inner_threads(jobs) <= cores + jobs
        assert cli._inner_threads(jobs) >= 1


def test_nested_fanout_is_inside_the_same_core_budget():
    """The bug this exists to prevent: budgeting on `pair_jobs` alone.

    Stages 03/05/06/07 open a second `Parallel` INSIDE each unit, so the leaf
    count is the product of the two levels, not the outer one. Observed in the
    wild: `--pair-jobs 3 --n-jobs 6` produced 3 outer workers at 10 BLAS threads
    each PLUS 18 nested processes -- 48 runnable threads on 32 cores -- because
    `_inner_threads` had never been told the inner pool existed.
    """
    cores = os.cpu_count() or 4
    for jobs in (1, 2, 3, 5, 8):
        for fan in (1, 2, 6, 16, 64):
            pj, inner, thr = cli.plan_concurrency(jobs, fan)
            assert inner == fan
            assert thr >= 1
            if jobs * fan <= cores:
                assert jobs * fan * thr <= cores


def test_auto_inner_jobs_splits_what_pair_jobs_left():
    cores = os.cpu_count() or 4
    for jobs in (1, 2, 4):
        pj, inner, thr = cli.plan_concurrency(jobs, "auto")
        assert pj * inner * thr <= cores


def test_minus_one_no_longer_means_every_core():
    """joblib reads -1 as `cpu_count`; under an outer pool that is the pile-up.

    Three stages defaulted to `--n-jobs -1`, so `--pair-jobs 3` asked for 96
    processes on a 32-core box. -1 is still accepted, but means `auto`.
    """
    cores = os.cpu_count() or 4
    assert cli.resolve_inner_jobs(-1, 3) == cli.resolve_inner_jobs("auto", 3)
    assert cli.resolve_inner_jobs(-1, 3) < cores


def test_an_explicit_inner_width_is_honoured_verbatim():
    """Deliberate oversubscription stays available for I/O-bound sub-units."""
    assert cli.resolve_inner_jobs(64, 4) == 64


def test_stages_that_fan_out_internally_expose_n_jobs():
    ap = cli.base_parser("x", inner_jobs=True)
    assert ap.parse_args([]).n_jobs == "auto"
    assert not hasattr(cli.base_parser("x").parse_args([]), "n_jobs")


def test_gpu_stages_default_to_one_pair_at_a_time():
    """Fanning out a GPU stage over pairs loads the model N times onto one card."""
    gpu = cli.base_parser("x", pair_jobs_default="1")
    assert gpu.parse_args([]).pair_jobs == "1"
    cpu = cli.base_parser("x")
    assert cpu.parse_args([]).pair_jobs == "auto"
