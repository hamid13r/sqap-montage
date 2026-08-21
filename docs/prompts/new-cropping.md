# Implementation prompt: simplify crop centering, guarantee output size

Branch to create: `new-cropping` (branch from `main`)
Primary file: `square_aperture_montage/crop_images.py`
Status: spec only — nothing has been implemented yet.

---

## 0. Branch setup

Another branch (`use-oldedge-frames`) has unrelated in-flight work on
`square_aperture_montage/blend_tiles.py`. **Do not touch that file.** Start clean:

```bash
git checkout main
git checkout -b new-cropping
```

If `.git/index.lock` or `.git/HEAD.lock` exist and no git process is actually running,
remove them first — they are stale.

---

## 1. What is wrong today

`detect_crop_boundaries()` in `square_aperture_montage/crop_images.py` currently does
this after finding the illuminated columns/rows (`x_lit`, `y_lit`):

```python
center_x = (x_lit[0] + trim + x_lit[-1] - trim) // 2
center_y = (y_lit[0] + trim + y_lit[-1] - trim) // 2

x_start = int(center_x - crop_x // 2)
x_end   = int(center_x + crop_x // 2)
...
# clamp each edge independently
x_start_c = max(0, x_start)
x_end_c   = min(int(width),  x_end)
y_start_c = max(0, y_start)
y_end_c   = min(int(height), y_end)
```

Three problems:

1. **`trim` is dead code.** `+ trim` and `- trim` cancel algebraically, so
   `center_x == (x_lit[0] + x_lit[-1]) // 2` regardless of the value. The docstring
   claims it "trims extra fringe pixels"; it does not. The parameter is exposed all the
   way out to the CLI and the YAML config, so it is actively misleading.
2. **The clamp truncates the window instead of moving it.** When `x_start` or `y_start`
   comes out negative, the code chops the edge off rather than sliding the window back
   inside the image. The output MRC is then *smaller* than `crop_x × crop_y`, and the
   illuminated area is no longer centered in it — it sits displaced toward the clamped
   edge. This is a real, observed bug: with `ImageSize = 5760 4092` and a 3840×3840 crop
   there are only **126 px of Y margin per side**, so `y_start` goes negative on
   essentially every tile and every cropped image comes out shifted.
3. **Floor division biases the center.** `// 2` always rounds down, so a half-pixel
   center lands consistently toward lower x/y — the same direction as bug (2).

---

## 2. The algorithm you must implement

Replace the centering logic with exactly this, and nothing more elaborate:

1. Find the illuminated extent: `x_min, x_max = x_lit[0], x_lit[-1]` and
   `y_min, y_max = y_lit[0], y_lit[-1]`.
2. Find the center of the illuminated area:
   `center_x = (x_min + x_max) / 2`, `center_y = (y_min + y_max) / 2`.
   Keep this as a float and round **once** at the end — do not floor-divide.
3. Cut a `crop_x × crop_y` window centered on that point:
   `x_start = int(round(center_x - crop_x / 2))`, `x_end = x_start + crop_x`
   (derive `x_end` from `x_start + crop_x`, **not** from `center + crop_x/2` — that
   guarantees the width is exactly `crop_x` for odd sizes too). Same for Y.
4. **The output must always be exactly `crop_x × crop_y`.** If the window spills past an
   edge, translate it back inside the image; do not truncate it:

   ```python
   if x_start < 0:
       x_start, x_end = 0, crop_x
   elif x_end > width:
       x_start, x_end = width - crop_x, width
   ```

   Same for Y against `height`. Emit a `[WARNING]` in the existing style when a
   translation happens, reporting how far the window moved — that displacement is
   scientifically meaningful (the crop is no longer centered on the illumination) and the
   user needs to see it.
5. **If the requested crop does not fit at all** (`crop_x > width` or
   `crop_y > height`), `raise ValueError` with a clear message naming both sizes. Do not
   silently emit an undersized tile — downstream `blendmont` requires uniform piece
   sizes, and a wrong-size tile is far worse than a hard failure.

Keep everything above the centering logic exactly as it is: the `float64` cast, the
`col_sum`/`row_sum` profiles, the `np.convolve` smoothing, the `mask_threshold`
selection, and both existing `ValueError` guards. They are sound and are not in scope.

---

## 3. Remove `trim` entirely

The user has confirmed trimming is not wanted. Remove the parameter — do not keep it as
an accepted no-op. It appears in these places (`build/` and `.egg-info/` excluded, see
§4.1):

| File | Lines | What |
| --- | --- | --- |
| `square_aperture_montage/crop_images.py` | 66 | `detect_crop_boundaries` signature |
| | 70 | docstring "trims extra fringe pixels" |
| | 118–119 | the centering expressions |
| | 151 | `crop_average` signature |
| | 165 | call into `detect_crop_boundaries` |
| | 259, 263, 268 | `_crop_image_worker` docstring tuple, unpack, and call |
| | 293–294 | `@click.option('--trim', ...)` |
| | 307, 332 | `main()` signature and call |
| `square_aperture_montage/run_pipeline.py` | 70 | `DEFAULT_CONFIG["crop"]["trim"]` |
| | 266 | `trim=c["trim"]` — **bracket access, will raise `KeyError` if you remove the default but leave this line.** Delete both together. |
| `sqap_montage.py` | 141 | `c.get('trim', 50)` |
| | 166, 182 | worker args tuple and the `crop_average` call |
| | 628 | embedded config template comment |
| `pipeline.yaml` | 71 | `trim: 50 # pixels to shave inside the detected edge` |

**Positional-tuple hazard.** `_crop_image_worker` takes a positional args tuple built in
two different places (`crop_images.py` `main()` is keyword-based, but `sqap_montage.py`
line 166 builds the tuple positionally). Removing an element means updating the
**docstring listing**, the **unpack**, and **every construction site** in lockstep. Grep
for `_crop_image_worker` and verify each one before finishing.

**Backward compatibility:** a user's existing `pipeline.yaml` may still contain
`trim: 50`. Both config readers use `.get()` on a per-key basis rather than validating
against a schema, so a leftover key is silently ignored and old configs keep working. No
migration shim needed — but do confirm this is still true after your edit.

---

## 4. Conventions to respect

### 4.1 Do not edit generated or archived copies

Stale duplicates exist. Edit only the file under `square_aperture_montage/`.

- ❌ `build/lib/square_aperture_montage/crop_images.py` — build artefact
- ❌ `square_aperture_montage.egg-info/` — generated
- ❌ `archive/` — historical
- ❌ `square_aperture_montage/blend_tiles.py` — in-flight work on another branch
- ✅ `square_aperture_montage/crop_images.py`

### 4.2 Config surfaces stay in sync

Any option change must land on all four surfaces: `crop_images.py` (click options),
`run_pipeline.py` (`DEFAULT_CONFIG` + `run_crop`), `sqap_montage.py` (config reads +
embedded template near line 620), and `pipeline.yaml`. This repo has a history of drift
between them — `snap_shifts_to_grid` never made it into `pipeline.yaml`, for instance.
Do not add to that.

### 4.3 Extend the boundary file for diagnosability

`crop_average()` writes `processing/crop/{stem}_crop_boundaries.txt`. Nothing in the repo
reads it (verified — it is write-only QC output), so it is safe to extend. Change it to:

```
x_start,x_end,y_start,y_end,x_min,x_max,y_min,y_max,center_x,center_y,shifted
```

where `x_min…y_max` are the detected illumination extents, `center_x/center_y` are the
computed illumination center, and `shifted` is a bool flag set when §2 step 4 had to
translate the window. This is what makes a systematic crop offset visible at a glance
next time instead of requiring a source read. Return the extra values from
`detect_crop_boundaries` — a small dataclass or a `NamedTuple` is cleaner than a 10-tuple;
your call, but keep the existing 4-value unpack working at the `crop_average` call site
or update it cleanly.

### 4.4 Version bump

Every functional commit in this repo bumps `pyproject.toml`. Bump the patch version.

---

## 5. Verification

Add `tests/test_crop_centering.py` using `pytest` and synthetic numpy arrays — no MRC
files, no IMOD, no real data:

1. **Centered aperture** — a bright square centered in a 5760×4092 field. Assert the
   returned window is exactly `crop_x × crop_y` and that the illumination center lands
   within 1 px of the output center.
2. **Off-center aperture, fits** — bright square offset a few hundred px. Same two
   assertions. This is the core regression test.
3. **Aperture near the bottom edge (the real bug)** — construct the case where
   `y_start` would go negative with a 3840 crop in a 4092-tall field. Assert the output
   is *still* exactly 3840×3840, that `y_start == 0`, and that the warning fires. Under
   the old code this returned an undersized array; that is the behaviour being fixed.
4. **Aperture near the top/right edges** — mirror of (3), asserting `y_end == height`
   and `x_end == width`.
5. **Crop larger than image** — `crop_y = 5000` on a 4092-tall field must raise
   `ValueError`, not clamp.
6. **`trim` is gone** — `inspect.signature(detect_crop_boundaries)` has no `trim`
   parameter, and `sam-crop --help` does not mention `--trim`.
7. **Output size invariant, parametrized** — sweep the aperture center across a grid of
   positions (including all four corners) and assert the returned window is
   `crop_x × crop_y` in every single case. This is the invariant that the old code
   violated.

Also confirm manually:

- `sam-crop --help` renders without `--trim`.
- `python sqap_montage.py write-config /tmp/t.yaml` emits a `crop:` block with no `trim`
  key, and that file round-trips through the crop step's config reader.
- `python sqap_montage.py crop --config pipeline.yaml --dry-run` (or the equivalent) runs
  without a `KeyError` on a config that still contains a stale `trim:` key.

Do not run the pipeline on real data — the user will validate the scientific result.

---

## 6. Out of scope

- Any change to `blend_tiles.py` (another branch owns it).
- Changing `crop_x`/`crop_y` defaults. Note in the PR description that with
  `ImageSize = 5760 4092` a 3840 Y-crop leaves only 126 px of margin per side, so
  translation warnings will be common and the user may want a smaller `crop_y` — but do
  not change the default here.
- Reworking the profile-smoothing or thresholding logic.
- Sub-pixel or interpolated cropping. Integer slicing only.
