"""csx_common — the shared base: paths, pair registry, cohorts, frozen constants.

Imported by all three components. Deliberately tiny and dependency-light (yaml +
pathlib only, no numpy at import time, never torch), because its whole job is to
let the three packages agree on *what exists* without depending on each other.

Anything that involves fitting, extraction or aggregation belongs in a component,
not here.
"""
