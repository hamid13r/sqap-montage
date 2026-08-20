# Implementation prompt: reuse the averages' blendmont solution when blending frames

Branch: `use-oldedge-frames`
Target file (primary): `square_aperture_montage/blend_tiles.py`
Status: spec only — nothing has been implemented yet.

---

## 1. Background: what the code does today

In `square_aperture_montage/blend_tiles.py`, `_blend_tilt_worker()` handles one tilt
angle. It does two things in sequence:

1. **Averages** — builds `image_list` from `cropped_averages_abs`, writes a `.plin`,
   then runs `newstack` → `blendmont` → `clip` producing
   `blended/averages/{ts}_{tilt_angle}_blended.mrc`.
2. **Frames** (`if blend_frames:`) — for each `frame_i in range(num_frames)`, rebuilds
   `frame_image_list` from `cropped_frames_abs`, writes a **separate** `.plin`, and runs
   its own independent `newstack` → `blendmont` → `clip`.

The only thing shared between the two is `shifts_list`, the per-tile
`PixelShiftFromCenter` values read from the mdoc (already snapped by
`_snap_shifts_to_uniform` when `snap_shifts_to_grid` is on).

### The problem

`imod_blendmont()` derives its edge-function root from the output filename:

```python
rootname = Path(blended_output).stem
...
f"-roo {os.path.join(processing_dir, rootname)} "
```

So the averages run uses root `{ts}_{tilt}_blended` and each frame uses root
`{ts}_{tilt}_frame{i}_blended` — all distinct. Combined with `-adj -shift` and no
`-oldedge`, **every frame recomputes its own edge functions and its own
cross-correlation displacements from scratch.**

Consequences:

- A 4-frame tilt runs **5 independent blendmont solutions**, each cross-correlating
  much noisier data than the average. Frames can land on slightly different edge
  alignments than the average and than each other.
- Edge correlation is the expensive part of blendmont, so this is also the dominant
  cost of the frame path.
- The averages' `.plout` is written and never read back.

### Relevant blendmont options (IMOD 5.2.0 man page, verified)

| Option | Long form | Behaviour |
| --- | --- | --- |
| `-roo` | `RootNameForEdges` | Root name for edge-function **and** `.ecd` files. Creates/seeks `ROOT.xef` and `ROOT.yef`. |
| `-oldedge` / `-ol` | `OldEdgeFunctions` | "Use existing edge functions, if they exist, rather than computing new ones." Note the *if they exist* — it silently falls back, it does not error. |
| `-readxcorr` / `-re` | `ReadInXcorrs` | "Read displacements in the overlap zones from an existing `.ecd` file instead of computing correlations." |
| `-shift` / `-sh` | `ShiftPieces` | Already passed. Default is to use both edge functions and cross-correlations per section and pick the lower-error one — which is why a `.ecd` gets written today. |

The man page carries this caveat: *"if the input image file is changed in any way, the
edge functions must be recalculated."* That warning is about changes to piece
geometry/binning/gradient correction, not about blending a different but
identically-tiled exposure of the same field. Reusing the average's solution for its own
frames is the intended kind of reuse — but it **must be opt-in and testable against real
data**, which is the entire point of this branch.

---

## 2. Goal

Add a **three-way** option controlling how much of the averages' blendmont solution the
frame blends reuse.

**Name:** `frame_edge_reuse`
**CLI:** `--frame-edge-reuse [none|edges|edges-xcorr]`
**Default:** `none` — current behaviour must be byte-for-byte unchanged when the option
is not set.

| Value | Extra blendmont flags on the frame runs | Meaning |
| --- | --- | --- |
| `none` | *(none)* | Today's behaviour. Every frame solves independently. |
| `edges` | `-oldedge` | Reuse the averages' `.xef`/`.yef`; each frame still recomputes its own correlation displacements. |
| `edges-xcorr` | `-oldedge -readxcorr` | Reuse the averages' `.xef`/`.yef` **and** `.ecd`. Frames inherit the average's geometry exactly. Maximum consistency, biggest speedup. |

This is deliberately a 3-way choice, not a boolean, so all three can be compared
empirically on real tilt-series before picking a default.

---

## 3. How to wire the edge-file reuse

### Do NOT simply point the frames' `-roo` at the averages' root

Tempting, but wrong for two reasons: blendmont may rewrite `.xef`/`.yef`/`.ecd` under
that root, destroying the pristine averages solution; and it would write files from
`processing/blending_frames/` work into `processing/blending_averages/`, breaking the
directory separation the rest of the code relies on.

### Instead: copy, then reuse

Before each frame's blendmont call, when `frame_edge_reuse != 'none'`, copy the
averages' edge files to the frame's own root:

```
{proc_avg}/{ts}_{tilt}_blended.xef  →  {proc_frm}/{ts}_{tilt}_frame{i}_blended.xef
{proc_avg}/{ts}_{tilt}_blended.yef  →  {proc_frm}/{ts}_{tilt}_frame{i}_blended.yef
{proc_avg}/{ts}_{tilt}_blended.ecd  →  {proc_frm}/{ts}_{tilt}_frame{i}_blended.ecd   # only for edges-xcorr
```

Then run the frame's blendmont with its existing `-roo` plus `-oldedge`
(and `-readxcorr` for `edges-xcorr`).

This keeps `processing/blending_frames/` self-contained, leaves the averages' solution
untouched as the reference, and means each frame's actual inputs are inspectable
after the fact.

### Ordering

`_blend_tilt_worker` already runs the averages blend before the frame loop, sequentially
within the worker, so the edge files exist by the time the frames need them. Tilt angles
run in separate processes but each has its own root, so there is no cross-process
collision. **No change to the concurrency model is needed — do not restructure it.**

### Fall back, don't crash

If an expected `.xef`/`.yef`/`.ecd` is missing (e.g. the averages blendmont failed, or
blendmont wrote a `.xecd`/`.yecd` pair instead of a `.ecd` — see the
`-functions`/`EdgeFunctionsOnly` note in the man page), emit a
`[WARNING]` in the same style as the existing warnings in this file and fall back to
computing that frame from scratch. A missing edge file must never abort the run.

---

## 4. Watch out for: file and code conventions

These are the things most likely to be got wrong. Please respect all of them.

### 4.1 Do not edit generated or archived copies

There are **stale duplicates of `blend_tiles.py`** in the repo. Edit only the one under
`square_aperture_montage/`:

- ❌ `build/lib/square_aperture_montage/blend_tiles.py` — build artefact, do not touch
- ❌ `square_aperture_montage.egg-info/` — generated, do not touch
- ❌ `archive/make_montages_mdoc_square_frames_mrc.py` — historical, do not touch
- ✅ `square_aperture_montage/blend_tiles.py`

### 4.2 Thread the option through all four config surfaces

`snap_shifts_to_grid` is the precedent to copy — follow exactly the same pattern. It
touches:

1. **`square_aperture_montage/blend_tiles.py`**
   - `imod_blendmont()` — new keyword params (see §5)
   - `_blend_tilt_worker()` — the positional args tuple: update the **docstring tuple
     listing**, the **unpack**, and the **construction site** in `process_tilt_series`.
     Append the new element at the end, matching how `snap_shifts_to_grid` was added.
   - `process_tilt_series()` — new keyword param + numpydoc entry in the docstring
   - `main()` — new `@click.option`
2. **`square_aperture_montage/run_pipeline.py`**
   - `DEFAULT_CONFIG["blend"]` (~line 100) — new key with an explanatory comment
   - `run_blend()` (~line 288) — `c.get("frame_edge_reuse", "none")`
   - the `process_tilt_series(...)` call (~line 337) — pass it
3. **`sqap_montage.py`**
   - the blend command's config reads (~line 275) — `b.get('frame_edge_reuse', 'none')`
   - the `process_tilt_series(...)` call (~line 328) — pass it
   - the **embedded config template** (~line 655, in the `blend:` block) — add the key
     with comments
4. **`pipeline.yaml`**
   - add the key to the `blend:` section with a comment block in the existing style

**Known drift — do not replicate it.** Two options are currently missing from some of
these surfaces: `snap_shifts_to_grid` is absent from `pipeline.yaml`, and
`sqap_montage.py`'s blend command never passes `normalize_averages_to_center` /
`normalize_frames_to_center` through to `process_tilt_series`. Add `frame_edge_reuse` to
**all four** surfaces. Fixing the two pre-existing gaps is welcome but should be a
**separate commit** on this branch so the feature diff stays reviewable.

### 4.3 Keep `{ts}.sh` re-runnable

`process_tilt_series` writes `processing/sh_files/{ts}.sh` from the `commands` list
returned by the worker, and advertises it as re-runnable by hand. The edge-file copies
happen in Python, so a shell script that only contains the IMOD commands would no longer
reproduce the run.

**Requirement:** append the copies to the `commands` list as literal `cp` lines
(e.g. `cp -f "/…/{ts}_{tilt}_blended.xef" "/…/{ts}_{tilt}_frame0_blended.xef"`),
in the same position in the sequence where they actually run. Use the same quoting the
script needs — tilt angles are floats, so the generated filenames contain `.` and `-`
characters (`VLP3x3_p04_ts_004_-60.0_blended.xef`). Quote the paths.

### 4.4 Preserve existing filename conventions

Do not change how any existing path is constructed. The `{ts}_{tilt_angle}` stem (with
its float formatting) appears in `.plin`, `.plout`, `.mrc`, `.log`, and `.sh` names and
is load-bearing across steps. This change adds files with that stem; it must not rename
any.

### 4.5 Cleanup interaction

`_cleanup_dir()` in `sqap_montage.py` deletes only `**/*.mrc` and leaves everything else,
so `.xef`/`.yef`/`.ecd` survive a run — which is what we want. Verify this still holds;
do not extend the cleanup glob.

### 4.6 Logging

`imod_blendmont` already writes a per-command log via `_write_command_log`, and the
recorded command string will automatically show the new flags. Additionally log, once
per tilt at INFO/print level, which reuse mode was actually applied (including any
fallback to `none`), so a test run can be audited from the logs.

### 4.7 Version bump

Every functional commit in this repo's history bumps `pyproject.toml`. Bump
`version = "0.1.5"` → `"0.1.6"`.

---

## 5. Suggested signature changes

Keep these additive and keyword-only-with-defaults so nothing existing breaks.

```python
FRAME_EDGE_REUSE_CHOICES = ('none', 'edges', 'edges-xcorr')

def imod_blendmont(stk_file, plin_file, plout_file, blend_size,
                   blended_output, processing_dir,
                   blend_log_path=None, clip_log_path=None,
                   old_edge=False, read_xcorr=False):
    ...
```

Build the flag suffix from `old_edge` / `read_xcorr` and append to `blend_cmd`. Leave
`-adj -shift` in place in all modes. Do **not** change how `rootname`, `intermediate`,
or the `-al {plout_file}` argument are derived.

```python
def process_tilt_series(..., snap_shifts_to_grid=True,
                        frame_edge_reuse='none'):
```

Validate the value early (in `main()` via `click.Choice(FRAME_EDGE_REUSE_CHOICES)`, and
defensively in `process_tilt_series` for the library-call path) and raise a clear error
on an unknown string rather than silently treating it as `none`.

---

## 6. Verification

There is no test suite in the repo yet, and IMOD is not available in CI, so:

1. Add `tests/test_blend_commands.py` with `pytest` tests that monkeypatch
   `subprocess.run` (returning a stub with `returncode=0`, `stdout=b''`, `stderr=b''`)
   and assert on the **generated command strings**:
   - `frame_edge_reuse='none'` produces frame blendmont commands with neither
     `-oldedge` nor `-readxcorr` — and the full command string is **identical** to what
     the current `main` branch produces (this is the regression guard).
   - `frame_edge_reuse='edges'` adds `-oldedge` only, and only to the **frame** calls —
     the averages call must be unchanged.
   - `frame_edge_reuse='edges-xcorr'` adds both `-oldedge` and `-readxcorr`.
   - The `cp` lines appear in the returned `commands` list in the right order.
   - A missing `.xef` triggers the warning path and falls back rather than raising.
2. Confirm `sam-blend --help` and `python sqap_montage.py blend --help` render the new
   option.
3. Confirm `python sqap_montage.py write-config /tmp/test.yaml` emits the new key, and
   that the emitted file round-trips through the blend step's config reader.
4. Run a `--dry-run` blend against `pipeline.yaml` to confirm the key is picked up.

Do not attempt to run real IMOD commands — the user will validate the scientific result
on real tilt-series after review.

---

## 7. Out of scope for this branch

Mention these in the PR description if you like, but do not implement them here:

- Converting the `_blend_tilt_worker` positional args tuple to a dataclass or dict
  (worth doing, but it would swamp this diff).
- Reusing the averages' `.plout` for anything.
- Changing the default away from `none`.
- Parallelising the frame loop.
