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

## Configuration

`pipeline.yaml` controls every parameter. The key sections are:

```yaml
data_dir: /path/to/your/dataset   # all relative paths are resolved from here

crop:
  input_dir:   frames/averages    # motion-corrected average MRCs
  crop_x:      3840               # final tile size (pixels)
  crop_y:      3840

blend:
  mdoc_dir:    mdocs              # per-tile SerialEM .mrc.mdoc files
  blend_size:  11664              # output image edge length after stitching
  num_frames:  4                  # raw frames per exposure

fill:
  gpus: "0"                       # GPU IDs, or "cpu"

make_mdoc:
  dose_per_tilt: 3.0              # e-/Å² per tilt
```

See the inline comments in `pipeline.yaml` for all options.

---

## Directory layout after a full run

```
<data_dir>/
  mdocs/                   input: per-tile SerialEM .mrc.mdoc files
  frames/averages/         input: motion-corrected average MRCs
  frames/                  input: raw per-exposure frame stacks
  cropped/averages/        step 1 output
  cropped/frames/
  blended/averages/        step 2 output (blended averages + mdocs)
  blended/frames/
  blended/frames_filled/   step 3 output (gap-filled frames)
  blended_mdocs/           step 4 output (combined .mrc.mdoc per tilt-series)
  processing/              intermediate files (safe to delete after pipeline)
```

---

## License

MIT — see `pyproject.toml`.
