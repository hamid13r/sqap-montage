#!/usr/bin/env python3
"""ts_filter.py — shared tilt-series filtering used by every pipeline step.

``ts_filter`` is a global config option: a list of tilt-series name patterns.
Each pattern is a shell-style glob (``fnmatch``), so ``*``, ``?`` and ``[..]``
are supported. An empty list means "process everything".

Two matching modes are needed because the steps operate on different units:

* **Tilt-series steps** (blend, make-mdoc) discover tilt-series *names* from the
  mdoc files (e.g. ``VLP3x3_p04_ts_004``). :func:`ts_matches` matches a name
  against the patterns with an anchored ``fnmatch`` — so a plain name behaves as
  an exact match (backward compatible) and ``VLP3x3_p04_ts_*`` works as a glob.

* **File steps** (crop, fill) operate on individual ``.mrc`` files whose names
  *embed* the tilt-series name behind an acquisition timestamp
  (``2025-02-17_..._VLP3x3_p04_ts_004_0_0_...mrc``). :func:`path_matches` treats
  each pattern as a substring (``*pattern*``) of the basename, so the same
  ``ts_filter`` selects the right files even though the name is not a prefix.
"""

import fnmatch
import os


def normalize_patterns(patterns):
    """Coerce a config value into a list of pattern strings.

    Accepts None (→ []), a single string (→ [string]), or a list. This keeps
    both ``ts_filter: VLP3x3_p04_ts_004`` and ``ts_filter: [a, b]`` working.
    """
    if not patterns:
        return []
    if isinstance(patterns, str):
        return [patterns]
    return [str(p) for p in patterns]


def ts_matches(name, patterns):
    """True if *name* matches any pattern (anchored glob). [] → match all."""
    pats = normalize_patterns(patterns)
    if not pats:
        return True
    return any(fnmatch.fnmatch(name, p) for p in pats)


def path_matches(path, patterns):
    """True if the basename of *path* contains a glob match for any pattern.

    [] → match all. Used for file-based steps where the tilt-series name is
    embedded in a longer filename.
    """
    pats = normalize_patterns(patterns)
    if not pats:
        return True
    base = os.path.basename(path)
    return any(fnmatch.fnmatch(base, f"*{p}*") for p in pats)


def filter_names(names, patterns):
    """Return the subset of tilt-series *names* matching *patterns*."""
    pats = normalize_patterns(patterns)
    if not pats:
        return list(names)
    return [n for n in names if ts_matches(n, pats)]


def filter_paths(paths, patterns):
    """Return the subset of file *paths* whose basename matches *patterns*."""
    pats = normalize_patterns(patterns)
    if not pats:
        return list(paths)
    return [p for p in paths if path_matches(p, pats)]
