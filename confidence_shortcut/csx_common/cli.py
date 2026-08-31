"""Shared CLI scaffolding.

Two conventions, both deliberate:

**No default action.** Every stage requires an explicit `--plan` or `--run`.
Running with no flags prints the plan and exits non-zero. Extraction stages cost
GPU-hours and probe stages write into a shared store, so "I just wanted to see
what it would do" must never be able to start work by accident. This mirrors the
existing repo convention (`--stats` / `--run` / `--report` in scripts/dse_splits).

**`--plan` is honest about cost.** It lists the work units, the inputs each one
reads, the outputs it would write, and what is already done and would be skipped.
That is what makes a long extraction schedulable rather than a leap of faith.

**Units run in parallel by default.** A unit is one pair, every unit writes its
own checkpoint, and no unit reads another's output -- so the batch is
embarrassingly parallel and running it as a for-loop wastes most of the machine.
`--pair-jobs` sets the width; `auto` picks it from the core count. The inner
thread pool is divided down to match, because `pair_jobs` processes each
grabbing every core is slower than the sequential loop it replaced, not faster.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass
class Unit:
    """One resumable piece of work."""
    key: str
    outputs: Sequence[Path] = ()
    inputs: Sequence[Path] = ()
    note: str = ""
    cost: str = ""          # free-text estimate, e.g. "15k rows, eager attn"
    # Preconditions that are NOT expressible as "does this file exist": phase 2
    # needs rows.parquet to exist AND to have its spans filled in by phase 1.
    # Without this, --plan reports such a unit as ready and it then fails at
    # runtime, which defeats the point of having a plan mode at all.
    unmet: Sequence[str] = ()

    @property
    def done(self) -> bool:
        return bool(self.outputs) and all(Path(o).exists() for o in self.outputs)

    @property
    def blocked(self) -> list[str]:
        missing = [f"missing {i}" for i in self.inputs if not Path(i).exists()]
        return missing + list(self.unmet)


def base_parser(description: str, *,
                pair_jobs_default: str = "auto",
                inner_jobs: bool = False) -> argparse.ArgumentParser:
    """Standard flags. GPU stages pass `pair_jobs_default="1"`.

    A GPU stage must not fan out over pairs on its own: each worker would load
    its own copy of the model onto the same device, so the default that is right
    for a CPU stage is an out-of-memory crash for an extraction one. Those stages
    are parallelised by launching one process per GPU instead.

    `inner_jobs=True` adds `--n-jobs` for stages that ALSO fan out inside a unit.
    Those stages must pass the resolved value to `run_units(inner_fanout=...)`,
    or the two levels of parallelism multiply instead of sharing the machine --
    see `plan_concurrency`.
    """
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true",
                      help="list the work units and exit; touches nothing")
    mode.add_argument("--run", action="store_true",
                      help="actually do the work")
    ap.add_argument("--pairs", default=None,
                    help="comma-separated pair keys (default: all active)")
    ap.add_argument("--force", action="store_true",
                    help="redo units whose outputs already exist")
    ap.add_argument("--include-pending", action="store_true",
                    help="include pairs whose data has not landed yet")
    ap.add_argument("--pair-jobs", default=pair_jobs_default,
                    help=f"units to run concurrently: an integer, or 'auto' "
                         f"to size from the core count (default: "
                         f"{pair_jobs_default})")
    if inner_jobs:
        ap.add_argument("--n-jobs", default="auto",
                        help="worker processes WITHIN one unit: an integer, or "
                             "'auto' to take an even share of what --pair-jobs "
                             "leaves free (default: auto). The two levels are "
                             "budgeted together, so raising this does not "
                             "oversubscribe the box.")
    return ap


def resolve_pair_jobs(spec: str | int, n_units: int) -> int:
    """How many units to run at once. Never more than there are units."""
    if n_units <= 1:
        return 1
    if str(spec) == "auto":
        # Two cores per unit: these are BLAS-bound fits, and one core per unit
        # leaves the machine idle whenever a unit is in a serial stretch.
        want = max(1, (os.cpu_count() or 4) // 2)
    else:
        want = max(1, int(spec))
    return min(want, n_units)


def _inner_threads(pair_jobs: int, inner_fanout: int = 1) -> int:
    """Threads each leaf worker's BLAS may use, so nothing oversubscribes.

    Oversubscription is the failure mode that makes parallelism look useless:
    `pair_jobs` processes x 32 OpenBLAS threads each is 5-10x more runnable
    threads than cores, and the resulting context-thrash can run slower than one
    process at full width.

    `inner_fanout` is the width of any pool a unit opens INSIDE itself. It is not
    optional bookkeeping: several stages (03, 05, 06, 07) run a nested `Parallel`
    over their own sub-units, so the real leaf count is the PRODUCT of the two
    levels. Budgeting on `pair_jobs` alone -- which this function used to do --
    silently permits `pair_jobs x n_jobs x threads` runnable threads, and with
    the old `--n-jobs -1` default that is 3 x 32 processes on a 32-core box.
    """
    cpu = os.cpu_count() or 4
    return max(1, cpu // max(1, pair_jobs * max(1, inner_fanout)))


def resolve_inner_jobs(spec: str | int, pair_jobs: int) -> int:
    """Width of the pool a unit opens inside itself, given the outer width.

    `auto` splits the machine evenly across the units already running, so the
    two levels of parallelism share the cores rather than each claiming all of
    them. An explicit integer is honoured as given -- including a deliberately
    oversubscribed one -- because a stage whose sub-units are I/O-bound or wildly
    uneven can legitimately want more workers than cores.

    `-1` is accepted for backwards compatibility and means `auto`, NOT
    `cpu_count`: joblib's own reading of -1 is what produced the 96-process
    pile-up this budgeting exists to prevent.
    """
    cpu = os.cpu_count() or 4
    if str(spec) in ("auto", "-1"):
        return max(1, cpu // max(1, pair_jobs))
    return max(1, int(spec))


def plan_concurrency(pair_jobs: int, inner_spec: str | int = 1) -> tuple[int, int, int]:
    """`(pair_jobs, inner_jobs, inner_threads)` whose product fits the machine.

    One place decides how a stage uses the box, so `--plan` can print the same
    numbers `--run` will use.
    """
    inner = resolve_inner_jobs(inner_spec, pair_jobs)
    return pair_jobs, inner, _inner_threads(pair_jobs, inner)


def require_mode(ap: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Resolve the action, refusing to guess."""
    if args.run:
        return "run"
    if args.plan:
        return "plan"
    ap.print_usage(sys.stderr)
    print("\nRefusing to guess: pass --plan to see the work, or --run to do it.",
          file=sys.stderr)
    raise SystemExit(2)


def print_plan(title: str, units: Iterable[Unit], *, force: bool = False) -> int:
    """Render the plan. Returns the number of units that would actually run."""
    units = list(units)
    todo, done, blocked = [], [], []
    for u in units:
        if u.blocked:
            blocked.append(u)
        elif u.done and not force:
            done.append(u)
        else:
            todo.append(u)

    print(f"=== {title} ===")
    print(f"{len(units)} unit(s): {len(todo)} to run, {len(done)} cached, "
          f"{len(blocked)} blocked\n")

    if todo:
        print("WOULD RUN")
        for u in todo:
            bits = [u.key]
            if u.cost:
                bits.append(f"[{u.cost}]")
            if u.note:
                bits.append(f"-- {u.note}")
            print("  " + " ".join(bits))
            for o in u.outputs:
                print(f"      -> {o}")
        print()
    if blocked:
        print("BLOCKED (unmet preconditions)")
        for u in blocked:
            print(f"  {u.key}")
            for m in u.blocked:
                print(f"      {m}")
        print()
    if done and not force:
        print(f"CACHED ({len(done)}): " + ", ".join(u.key for u in done[:12])
              + (" ..." if len(done) > 12 else ""))
        print()
    return len(todo)


def _run_one(fn: Callable[[Unit], str], u: Unit) -> tuple[str, bool, str]:
    """Run one unit, converting a failure into a value rather than a raise.

    Returned rather than raised so that one bad pair cannot cancel the pool and
    cost every other in-flight unit its work.
    """
    try:
        return u.key, True, fn(u)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return u.key, False, f"{type(exc).__name__}: {exc}"


def run_units(title: str, units: Iterable[Unit], fn: Callable[[Unit], str], *,
              force: bool = False, pair_jobs: str | int = "auto",
              inner_fanout: int = 1) -> int:
    """Execute units, skipping cached and blocked ones. Returns an exit code.

    `inner_fanout` MUST be passed by any stage whose unit function opens its own
    `Parallel`, so the thread budget is divided across both levels rather than
    handed out twice. Leaving it at 1 when a unit fans out is not a tuning
    mistake, it is an oversubscription bug -- and a quiet one, since the stage
    still completes, just several times slower than the loop it replaced.

    Failures are reported and counted rather than aborting the batch: a single
    bad pair should not cost the other twenty their progress, and per-unit
    checkpoints mean a rerun picks up exactly where this left off.

    Units run concurrently. Results are printed as they land, not in unit order,
    so a long batch still shows progress -- with `[1/6]` counters, since
    completion order no longer tells you how far along you are.
    """
    units = list(units)
    ok = failed = skipped = 0
    print(f"=== {title} ===", flush=True)

    todo: list[Unit] = []
    for u in units:
        if u.blocked:
            print(f"[skip] {u.key}: {u.blocked[0]}", flush=True)
            skipped += 1
        elif u.done and not force:
            print(f"[cached] {u.key}", flush=True)
            skipped += 1
        else:
            todo.append(u)

    n_jobs = resolve_pair_jobs(pair_jobs, len(todo))
    results: Iterable[tuple[str, bool, str]]
    if n_jobs == 1:
        results = (_run_one(fn, u) for u in todo)
    else:
        from joblib import Parallel, delayed, parallel_config
        inner = _inner_threads(n_jobs, inner_fanout)
        fan = (f" x {inner_fanout} inner worker(s)" if inner_fanout > 1 else "")
        print(f"    {len(todo)} units, {n_jobs} at a time{fan}, "
              f"{inner} thread(s) each "
              f"(<= {n_jobs * max(1, inner_fanout) * inner} of "
              f"{os.cpu_count()} cores)", flush=True)
        with parallel_config(backend="loky", inner_max_num_threads=inner):
            # generator_unordered so a finished unit prints immediately instead
            # of waiting behind a slower one that started earlier.
            results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                delayed(_run_one)(fn, u) for u in todo)

    for i, (key, good, msg) in enumerate(results, 1):
        tag = f"[{i}/{len(todo)}]"
        if good:
            ok += 1
            print(f"[ok] {tag} {key}{': ' + msg if msg else ''}", flush=True)
        else:
            failed += 1
            print(f"[FAIL] {tag} {key}: {msg}", flush=True)

    print(f"\n{ok} ok, {failed} failed, {skipped} skipped")
    return 1 if failed else 0
