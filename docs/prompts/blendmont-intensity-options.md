# Implementation prompt: expose blendmont intensity-correction options in the config

Branch to create: `blendmont-intensity-options` (branch from `main`)
Primary file: `square_aperture_montage/blend_tiles.py`
Also touched: `sqap_montage.py`, `square_aperture_montage/run_pipeline.py`, `pipeline.yaml`, `README.md`, `tests/`
Status: spec only — nothing has been implemented yet.

---

## 0. Branch setup

```bash
git checkout main
git pull
git checkout -b blendmont-intensity-options
```

`use-oldedge-frames` has unrelated in-flight work on `blend_tiles.py` — do **not** merge it or
cherry-pick from it. If `.git/index.lock` or `.git/HEAD.lock` exist and no git process is
running, they are stale; remove them.

---

## 1. Goal

`imod_blendmont()` in `square_aperture_montage/blend_tiles.py` currently hard-codes the
blendmont command:

```python
blend_cmd = (
    f"blendmont -imin {stk_file} -plin {plin_file} "
    f"-imout {intermediate} "
    f"-roo {os.path.join(processing_dir, rootname)} "
    f"-al {plout_file} -adj -shift"
)
```

blendmont's intensity-correction options are never used. The user must be able to turn them on
from the YAML config, per run, and have them appear in the generated blendmont command for both
the averages pass and the per-frame passes.

### The blendmont options to expose

From the blendmont man page (`INTENSITY CORRECTION OPTIONS`):

| blendmont flag | Type | Meaning |
|---|---|---|
| `-intensity N` (`-FixIntensityFromEdges`) | int | Solve for per-piece scaling factors that minimise intensity differences in the overlap zones. `2` first fits a planar gradient (averaged over the whole X and Y range) and adjusts the differences for it before solving. |
| `-base V` (`-BaseIntensityForScaling`) | float | Base value subtracted before scaling and added back after. Usually `0` for TEM; `-32768` if unsigned values are stored as signed ints. |
| `-sum` (`-SumPiecesForGradient`) | flag | Sum all pieces, run `clip planefit` on the input, read back the gradient slopes, and adjust every piece for that planar gradient as it is read in. |
| `-other FILE` (`-OtherSumGradientFile`) | path | Use a separately estimated planar gradient file. **Cannot be combined with `-sum`.** |
| `-flatfield FILE` (`-FlatfieldFile`) | path | Image to scale by, e.g. produced by `clip flatfield` (typically with `-n 2..4`). |

Important behavioural note to surface in the docs: **if `-sum` / `-other` / `-flatfield` is added
or dropped, the edge functions must be recomputed.** The pipeline writes edge functions to
`-roo <processing_dir>/<rootname>`, so stale `.ecd`/`.xef`/`.yef` files from a previous run with
different gradient settings would be silently reused. See §5.

---

## 2. Relationship to the existing normalization

`_normalize_tiles_to_center()` (mean/σ matching of off-centre tiles against the `_0_0` tile,
controlled by `normalize_averages_to_center` / `normalize_frames_to_center`) **stays exactly as
it is.** Do not remove, rename, or gate it behind the new options.

The two are independent and composable:

1. normalization (if enabled) rewrites the tile MRCs that go into the stack, then
2. blendmont's own intensity options act on that stack.

Both may be on at once. Say this explicitly in the config comments and the README.

---

## 3. Config schema

Add a nested `intensity` block under `blend:`. All keys optional; all defaults reproduce today's
behaviour exactly (no new flags emitted).

```yaml
blend:
  # ... existing keys ...

  # ---------------------------------------------------------------------------
  # blendmont intensity corrections
  # ---------------------------------------------------------------------------
  # These map directly onto blendmont's INTENSITY CORRECTION OPTIONS and are
  # appended to the generated blendmont command. They are independent of
  # normalize_*_to_center above — you can use either, both, or neither.
  intensity:
    # -intensity / -FixIntensityFromEdges
    #   0 = off (default), 1 = solve scaling factors from overlap-zone
    #   differences, 2 = also fit and remove a planar gradient first.
    fix_from_edges: 0

    # -base / -BaseIntensityForScaling
    #   Value subtracted before scaling and added back after. null = omit the
    #   flag (blendmont default). Use -32768 if unsigned data are stored as
    #   signed integers. Only meaningful with fix_from_edges > 0.
    base: null

    # -sum / -SumPiecesForGradient
    #   Sum all pieces, run "clip planefit" on the input, and correct every
    #   piece for the resulting planar gradient. Mutually exclusive with
    #   other_gradient_file.
    sum_for_gradient: false

    # -other / -OtherSumGradientFile
    #   Pre-computed planar gradient file. Mutually exclusive with
    #   sum_for_gradient. Relative paths resolve from data_dir.
    other_gradient_file: null

    # -flatfield / -FlatfieldFile
    #   Flatfield image to scale by, e.g. from "clip flatfield -n 3".
    #   Relative paths resolve from data_dir.
    flatfield_file: null
```

The same block must be added to **all three** places that define config, and they must stay
consistent:

1. `pipeline.yaml` (the checked-in example config)
2. `TEMPLATE_CONFIG` in `sqap_montage.py` (`write-config` output)
3. `DEFAULT_CONFIG` in `square_aperture_montage/run_pipeline.py`

---

## 4. Implementation

### 4.1 A pure builder function

In `blend_tiles.py`, add a small module-level function — pure, no I/O, easy to unit test:

```python
def build_intensity_args(fix_from_edges=0, base=None, sum_for_gradient=False,
                         other_gradient_file=None, flatfield_file=None):
    """Return the blendmont intensity-correction flags as a list of strings.

    Returns [] when nothing is enabled, so the generated command is byte-for-byte
    identical to the pre-existing one.
    """
```

Rules:

- `fix_from_edges` in `{0, 1, 2}`; `0` → emit nothing. Anything else → `ValueError`.
- `base` is `None` → omit; otherwise `-base {float(base):g}`.
- `sum_for_gradient` and `other_gradient_file` both set → `ValueError` naming both config keys.
- `base` set while `fix_from_edges == 0` → warn (it has no effect) but do not raise.
- File paths are quoted with `shlex.quote()` — the command is run through `shell=True`.
- Flag order: `-intensity`, `-base`, `-sum`, `-other`, `-flatfield` (deterministic, so tests and
  the logged `.sh` files are stable).

Add a companion `intensity_args_from_config(cfg_block, data_dir)` that reads the YAML block,
resolves `other_gradient_file` / `flatfield_file` against `data_dir` via the existing `resolve()`
helper, checks that each resolved file exists (raise `FileNotFoundError` with the config key name
if not), and returns the list. Validation must happen **once, up front**, before any tilt-series
is processed — not per tilt, and not silently at blendmont runtime.

### 4.2 Thread it through

`imod_blendmont(...)` gains a keyword-only `intensity_args=()` parameter appended to the command:

```python
blend_cmd = (
    f"blendmont -imin {stk_file} -plin {plin_file} "
    f"-imout {intermediate} "
    f"-roo {os.path.join(processing_dir, rootname)} "
    f"-al {plout_file} -adj -shift"
)
if intensity_args:
    blend_cmd += " " + " ".join(intensity_args)
```

Then plumb it (default `()` everywhere, so nothing existing breaks):

- `_blend_tilt_worker()` — the args tuple. It is pickled for the multiprocessing path, so
  **append the new element at the end of the tuple and update the unpacking, the docstring
  listing of the tuple layout, and every construction site together.** A mismatched tuple here
  fails at runtime inside a worker with a confusing traceback; grep for every place the tuple is
  built before you change it.
- `process_tilt_series()` — new `intensity_args=()` kwarg, passed to both the averages
  `imod_blendmont` call and the per-frame one, so averages and frames get identical corrections.
- `blend_tiles.py` `main()` CLI — add matching options so the module stays runnable standalone:
  `--intensity/-–fix-intensity-from-edges` (int, default 0), `--intensity-base` (float),
  `--sum-for-gradient/--no-sum-for-gradient`, `--other-gradient-file` (path),
  `--flatfield-file` (path).
- `square_aperture_montage/run_pipeline.py` `run_blend()` — build the args from
  `c.get("intensity", {})` and pass them; include the resolved flag string in the `--dry-run`
  output so `sam-run --dry-run` shows the exact blendmont flags.
- `sqap_montage.py` `blend()` — same.

### 4.3 Fix a related gap while you are here

`sqap_montage.py`'s `blend()` reads the config but **never passes
`normalize_averages_to_center` / `normalize_frames_to_center`** to `process_tilt_series()`, even
though `pipeline.yaml` documents both keys under `blend:`. Users setting them in the YAML and
running `python sqap_montage.py blend --config pipeline.yaml` silently get no normalization.
Read both keys and pass them through. Mention this fix in the PR description as a separate bullet
so it is not lost inside the feature.

---

## 5. Stale edge functions

blendmont writes edge functions to the `-roo` root inside `processing_dir` and reuses them if
present. Changing any gradient option invalidates them.

Implement the minimal safe thing: when the intensity settings are non-default, delete any
pre-existing edge-function files for that rootname (`*.ecd`, `*.xef`, `*.yef`) in
`processing_dir` before invoking blendmont, and log one line at INFO saying why. Do not add a
config knob for this. Add a note in the config comment and README that changing these options
invalidates cached edge functions.

---

## 6. Tests

New file `tests/test_blendmont_intensity.py`, pytest, no IMOD required:

1. Defaults → `build_intensity_args()` returns `[]`, and the assembled command string is
   identical to the current hard-coded one (guard against accidental whitespace changes).
2. `fix_from_edges=1` → `["-intensity", "1"]`; `=2` → `["-intensity", "2"]`; `=3` → `ValueError`.
3. `base=-32768` with `fix_from_edges=1` → `-base -32768` present, correct order.
4. `sum_for_gradient=True` **and** `other_gradient_file` set → `ValueError` mentioning both keys.
5. Paths with spaces are quoted.
6. `intensity_args_from_config` resolves relative paths against `data_dir` and raises
   `FileNotFoundError` (naming the config key) for a missing flatfield.
7. Monkeypatch `subprocess.run` in `imod_blendmont` and assert the captured command contains the
   flags exactly once, after `-shift`.
8. A regression test that `process_tilt_series(..., intensity_args=())` still produces the
   unchanged command — i.e. this feature is opt-in.

Run `pytest tests/ -q` and make sure the existing `test_crop_centering.py` still passes.

---

## 7. Docs

- `README.md`: short subsection under the blend step — the five options, one line each, the
  "these compose with `normalize_*_to_center`" note, the stale-edge-function warning, and one
  worked example (`fix_from_edges: 2` + `base: 0`).
- `examples/example_workflow.sh`: add a commented-out example only if it does not lengthen the
  script much.
- Config comments as in §3.

---

## 8. Verification before opening the PR

```bash
pytest tests/ -q
python sqap_montage.py write-config /tmp/fresh.yaml && diff <(sed -n '/^blend:/,/^fill:/p' /tmp/fresh.yaml) <(sed -n '/^blend:/,/^fill:/p' pipeline.yaml)
sam-run --config pipeline.yaml --steps blend --dry-run
python -c "import yaml,sys; yaml.safe_load(open('pipeline.yaml'))"
```

The three config sources must agree. Confirm the `--dry-run` output shows the blendmont flags,
and that with the block absent from a config the generated command is unchanged.

---

## 9. PR

```bash
git add -A
git commit -m "Expose blendmont intensity-correction options in the blend config"
git push -u origin blendmont-intensity-options
gh pr create --base main --title "Expose blendmont intensity-correction options" --body "..."
```

PR body should cover: what was added, that defaults are a no-op, that the existing
centre-tile normalization is untouched and composable, the `sqap_montage.py` normalization
pass-through fix, the stale-edge-function handling, and a note that the feature has **not** been
validated on real data yet — this branch is for testing.
