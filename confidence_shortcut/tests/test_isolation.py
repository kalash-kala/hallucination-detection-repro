"""The component boundary, enforced rather than merely intended.

The whole point of splitting extraction / probing / reporting into three packages
is that each can be built, reviewed and run without the others. That property
erodes silently the first time someone adds a convenience import, so it is
asserted here:

  * csx_probe and csx_report must import with torch absent -- they run on any
    CPU box, and a stray `import torch` would make them depend on a 2 GB wheel
    and a CUDA build for no reason.
  * csx_report must not reach into csx_probe or csx_extract; it consumes the
    results contract, nothing else.
  * csx_probe must never emit a `cohort` column -- cohorts are chosen at report
    time, and a cohort baked into an atomic table would defeat that.

csx_common is the one shared dependency all three may import: paths, the pair
registry, cohorts and the frozen constants. It exists precisely so the three
components can agree on *what exists* without importing each other.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "csx_probe": {"torch", "transformers", "PIL", "csx_extract", "csx_report"},
    "csx_report": {"torch", "transformers", "PIL", "csx_extract", "csx_probe"},
    "csx_extract": {"csx_probe", "csx_report"},
}


def _imported_top_levels(path: Path) -> set[str]:
    """Top-level module names imported by a source file, statically."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, i.e. within this same package
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def _sources(pkg: str) -> list[Path]:
    return sorted((ROOT / pkg).rglob("*.py"))


@pytest.mark.parametrize("pkg", sorted(FORBIDDEN))
def test_no_forbidden_imports(pkg: str):
    """Static scan: catches the violation even in a module that is never loaded
    at import time (a lazy `import torch` inside a function still shows up)."""
    offenders: dict[str, set[str]] = {}
    for src in _sources(pkg):
        bad = _imported_top_levels(src) & FORBIDDEN[pkg]
        if bad:
            offenders[str(src.relative_to(ROOT))] = bad
    assert not offenders, f"{pkg} must not import {FORBIDDEN[pkg]}: {offenders}"


@pytest.mark.parametrize("pkg", ["csx_probe", "csx_report"])
def test_imports_cleanly_without_torch(pkg: str, monkeypatch):
    """Dynamic check: import every submodule with torch masked out, so the CPU
    packages are proven to work on a box that has no torch installed at all."""
    class _Blocker:
        def find_module(self, name, path=None):
            return self if name.split(".")[0] in {"torch", "transformers"} else None

        def load_module(self, name):
            raise ImportError(f"{name} is blocked: {pkg} must not need it")

    for name in list(sys.modules):
        if name.split(".")[0] in {"torch", "transformers"}:
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])

    mod = importlib.import_module(pkg)
    for info in pkgutil.walk_packages(mod.__path__, prefix=f"{pkg}."):
        importlib.import_module(info.name)


def test_probe_never_reads_cohort_config():
    """csx_probe must not know which pairs form a group.

    This checks the COUPLING -- reading cohorts.yaml, or importing the cohort
    resolver -- rather than the spelling of the word. A module that mentions
    `cohort` only in order to *reject* the column is enforcing this rule, not
    breaking it, and an earlier version of this test failed exactly that guard.
    The emitted-column half of the property is asserted behaviourally below,
    which is stronger than a grep anyway: it catches a cohort column built under
    a name this scan never thought to look for.
    """
    offenders = {}
    for src in _sources("csx_probe"):
        text = src.read_text()
        hits = {tok for tok in ("cohorts.yaml", "csx_common.cohorts")
                if tok in text}
        if hits:
            offenders[str(src.relative_to(ROOT))] = sorted(hits)
    assert not offenders, (
        "csx_probe must stay cohort-blind; found references in: " f"{offenders}"
    )


def test_probe_refuses_to_write_a_cohort_column():
    """The behavioural half: an atomic table carrying a cohort is a hard error.

    Cohorts are chosen at report time. One baked into an atomic table would make
    re-grouping cost a refit, which is the entire reason the three components are
    split the way they are.
    """
    import pandas as pd
    import pytest

    from csx_probe import results

    df = pd.DataFrame({"pair": ["p"], "AUROC": [0.5], "cohort": ["qa8"]})
    with pytest.raises(ValueError, match="cohort"):
        results.write_unit("per_pair_long", "p", df)


def test_common_is_importable_by_everyone():
    """csx_common is the shared base -- it must not depend on any component."""
    offenders = {}
    for src in _sources("csx_common"):
        bad = _imported_top_levels(src) & {"csx_extract", "csx_probe", "csx_report",
                                           "torch", "transformers"}
        if bad:
            offenders[str(src.relative_to(ROOT))] = sorted(bad)
    assert not offenders, f"csx_common must depend on none of these: {offenders}"
