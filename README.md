# sqap-montage

**Square Aperture Montage** — an IMOD wrapper that stitches the 9 tilt-series tiles
collected by SerialEM's square-aperture montage acquisition mode into one large
tilt-series ready for downstream processing (e.g. AreTomo, CTFFIND, Relion).

SerialEM collects a 3 × 3 grid of overlapping tiles, each as a separate
motion-corrected stack. This pipeline:

1. **crop** — detects and removes the dark border outside the illuminated square aperture on each tile image
2. **blend** — calls IMOD `newstack` + `blendmont` + `clip resize` to stitch all 9 tiles per tilt angle into one large image
3. **fill** — fills the blending-seam artefacts with local texture using GPU-accelerated inpainting
4. **make-mdoc** — assembles a single `.mrc.mdoc` metadata file for the blended tilt-series, suitable for AreTomo or other downstream tools

---

## Requirements

| Dependency | Notes |
|---|---|
| [IMOD](https://bio3d.colorado.edu/imod/) | `newstack`, `blendmont`, `clip` must be on `$PATH` |
| Python ≥ 3.9 | via conda-forge or system Python |
| PyTorch (optional) | required only for the `fill` step |

---

## Installation

### Recommended: conda / micromamba (conda-forge)

```bash
# clone the repo
git clone https://github.com/hamid13r/sqap-montage.git
cd sqap-montage

# create the environment (installs all Python dependencies from conda-forge)
micromamba env create -f environment.yml
# or: conda env create -f environment.yml

micromamba activate sqap-montage
# or: conda activate sqap-montage

# install the package itself in editable mode
pip install -e .
```

For GPU support (PyTorch + cc3d for the `fill` step):

```bash
micromamba env create -f environment-gpu.yml
micromamba activate sqap-montage-gpu
pip install -e ".[gpu]"
```

### Pip only

```bash
pip install -e .
```

---

## Usage

All steps are driven by a single config file. Generate a template:

```bash
python sqap_montage.py write-config pipeline.yaml
```

Edit `pipeline.yaml` — at minimum set `data_dir` to your data directory and
`dose_per_tilt` in the `make_mdoc` section — then run each step:

```bash
python sqap_montage.py crop      --config pipeline.yaml
python sqap_montage.py blend     --config pipeline.yaml
python sqap_montage.py fill      --config pipeline.yaml   # requires PyTorch
python sqap_montage.py make-mdoc --config pipeline.yaml
```

If installed via `pip install -e .` the `sqap-montage` command is also available:

```bash
sqap-montage crop      --config pipeline.yaml
sqap-montage blend     --config pipeline.yaml
sqap-montage fill      --config pipeline.yaml
sqap-montage make-mdoc --config pipeline.yaml
```

---

## Selecting tilt-series (`ts_filter`)

`ts_filter` is a **global**, top-level config option that applies to **every**
step (crop, blend, fill, make-mdoc). It is a list of shell-style glob patterns
(`*`, `?`, `[..]` supported); leave it empty (`[]`) to process everything.

```yaml
# top level of pipeline.yaml
ts_filter: ["VLP3x3_p01_ts_*"]     # or exact: [VLP3x3_p01_ts_002, VLP3x3_p01_ts_003]
```

Matching differs by step because they operate on different units:

- **blend** and **make-mdoc** match the discovered **tilt-series name** (e.g.
  `VLP3x3_p04_ts_004`). A plain name is an exact match (backward compatible);
  add wildcards for ranges.
- **crop** and **fill** operate on individual `.mrc` files whose names *embed*
  the tilt-series name behind an acquisition timestamp, so a pattern matches
  when it appears **anywhere** in the filename.

The same `ts_filter` therefore selects a consistent set of tilt-series across
all four steps. (For backward compatibility a step-level `ts_filter` under
`blend`/`make_mdoc` is still honoured when the global one is unset.)

---

## blendmont intensity corrections (blend step)

The `blend` step can pass IMOD `blendmont`'s intensity-correction options
through from the config. Set them under `blend.intensity` in `pipeline.yaml`.
`fix_from_edges` **defaults to `1`** (solve per-piece scaling from the overlap
zones); set it to `0` to disable intensity correction entirely. The remaining
options default off. The flags are applied identically to the averages pass and
the per-frame passes.

| Config key | blendmont flag | Effect |
|---|---|---|
| `fix_from_edges` | `-intensity {1,2}` | Solve per-piece scaling factors from overlap-zone differences (`1`, the default); `2` also fits and removes a planar gradient first. `0` = off. |
| `base` | `-base {value}` | Value subtracted before scaling and added back after. Use `-32768` if unsigned data are stored as signed ints. Only meaningful with `fix_from_edges > 0`. |
| `sum_for_gradient` | `-sum` | Sum all pieces, run `clip planefit`, and correct every piece for the planar gradient. Mutually exclusive with `other_gradient_file`. |
| `other_gradient_file` | `-other {path}` | Use a pre-computed planar gradient file. Mutually exclusive with `sum_for_gradient`. |
| `flatfield_file` | `-flatfield {path}` | Flatfield image to scale by, e.g. from `clip flatfield -n 3`. |

Relative paths for `other_gradient_file` / `flatfield_file` resolve from
`data_dir` and are checked to exist once, up front, before any tilt-series is
processed.

These are **independent of and composable with** `normalize_averages_to_center`
/ `normalize_frames_to_center` — you can use either, both, or neither. The
center-tile normalization rewrites the tile MRCs that go into the stack;
blendmont's intensity options then act on that stack.

> **Note:** changing `sum_for_gradient` / `other_gradient_file` / `flatfield_file`
> invalidates blendmont's cached edge functions. sqap-montage deletes the stale
> `.ecd` / `.xef` / `.yef` files for the affected rootname automatically so they
> are recomputed.

Worked example — solve scaling *and* remove a planar gradient, with a base of 0:

```yaml
blend:
  intensity:
    fix_from_edges: 2
    base: 0
```

---

## Step-by-step walk-through
For a detailed walk-through of the pipeline, see [walk-through.md](walkthrough/walk-through.md).

## License

MIT — see `pyproject.toml`.
