#!/usr/bin/env python3
"""
run_pipeline.py — Orchestrator for the square-aperture montage pipeline.

Reads a YAML config file and runs any combination of the three pipeline steps:

  1. crop    — remove blank aperture borders  (crop_images.py)
  2. blend   — stitch tiles into one tilt-series (blend_tiles.py)
  3. fill    — fill blending-seam gaps          (remove_gaps.py)

Each step is called directly as a Python function (no subprocess), so errors
are caught and logged cleanly.

Usage
-----
  # Run all steps with default config
  sam-run --config pipeline.yaml

  # Only run crop + blend, skip gap filling
  sam-run --config pipeline.yaml --steps crop blend

  # Dry-run: print what would be done without doing it
  sam-run --config pipeline.yaml --dry-run

Generate a template config:
  sam-run --write-config pipeline.yaml
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import click

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Working directory — all relative paths are resolved from here.
    # Set to "." to use whatever directory you run the script from.
    "data_dir": ".",

    # Which steps to run (set any to false to skip)
    "steps": {
        "crop":  True,
        "blend": True,
        "fill":  True,
    },

    # ── Step 1: crop_images ──────────────────────────────────────────────────
    "crop": {
        "input_dir":       "frames/averages",
        "output_dir":      "cropped",
        "processing_dir":  "processing/crop",
        "frames_dir":      "frames",
        "crop_x":          3840,
        "crop_y":          3840,
        "filter_window":   200,
        "mask_threshold":  0.5,
        "trim":            50,
        "crop_frames":     True,
    },

    # ── Step 2: blend_tiles ──────────────────────────────────────────────────
    "blend": {
        "mdoc_dir":              "mdocs",
        "averages_dir":          "cropped/averages",
        "frames_dir":            "cropped/frames",
        "output_dir":            "blended",
        "processing_dir":        "processing",
        "blend_size":            11664,
        "num_frames":            4,
        "blend_frames":          True,
        # Rescale off-center average tiles to match center tile histogram before blending
        "normalize_averages_to_center": False,
        # Rescale off-center frame tiles to match center tile histogram before blending
        "normalize_frames_to_center":   False,
        # List specific tilt-series names to process, or leave empty for all:
        # ts_filter: [VLP3x3_p01_ts_002, VLP3x3_p01_ts_003]
        "ts_filter":             [],
    },

    # ── Step 3: remove_gaps ──────────────────────────────────────────────────
    "fill": {
        "input_dir":  "blended/frames",
        "output_dir": "blended/frames_filled",
        "mask_dir":   "blended/frames_masks",
        "gpus":       "0",      # comma-separated GPU IDs, or "cpu"
        "sigma":      5.0,
        "tile_num":   8,
        "resume":     True,
    },

    # ── Logging ──────────────────────────────────────────────────────────────
    "logging": {
        "log_dir": "logs",          # directory for log files; "" = don't write a file
        "level":   "INFO",          # DEBUG | INFO | WARNING | ERROR
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and merge a YAML config with defaults."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required to read config files.\n"
            "Install with: pip install pyyaml"
        )

    with open(config_path, 'r') as f:
        user_cfg = yaml.safe_load(f) or {}

    # Deep-merge user config over defaults
    cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_logging(log_dir: str, level_str: str, dry_run: bool) -> logging.Logger:
    """Configure root logger with console + optional file handler."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger = logging.getLogger("sqap")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir and not dry_run:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file  = os.path.join(log_dir, f"pipeline_{timestamp}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Logging to: {log_file}")

    return logger


def run_step(name: str, fn, kwargs: dict, logger: logging.Logger, dry_run: bool) -> bool:
    """Run one pipeline step, timing it and catching exceptions.

    Returns True on success, False on failure.
    """
    logger.info("=" * 60)
    logger.info(f"STEP: {name}")
    logger.info("=" * 60)

    if dry_run:
        logger.info(f"[DRY RUN] Would call {fn.__module__}.{fn.__name__} with:")
        for k, v in kwargs.items():
            logger.info(f"  {k}: {v}")
        return True

    t0 = time.time()
    try:
        fn(**kwargs)
        elapsed = time.time() - t0
        logger.info(f"✓ {name} completed in {elapsed:.1f}s")
        return True
    except SystemExit as exc:
        # click raises SystemExit(0) on --help, or non-zero on error
        if exc.code == 0:
            return True
        logger.error(f"✗ {name} exited with code {exc.code}")
        return False
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error(f"✗ {name} failed after {elapsed:.1f}s: {exc}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Step runners — translate config dict → function kwargs
# ---------------------------------------------------------------------------

def run_crop(cfg: dict, logger: logging.Logger, dry_run: bool) -> bool:
    from .crop_images import crop_average, crop_frames_for_image
    import glob
    from tqdm import tqdm

    c = cfg["crop"]

    if dry_run:
        logger.info("[DRY RUN] crop step:")
        for k, v in c.items():
            logger.info(f"  {k}: {v}")
        return True

    input_dir        = c["input_dir"]
    output_averages  = os.path.join(c["output_dir"], "averages")
    output_frames    = os.path.join(c["output_dir"], "frames")
    processing_dir   = c["processing_dir"]
    crop_frames_flag = c["crop_frames"]
    frames_dir       = c["frames_dir"]

    if os.path.isdir(input_dir):
        image_files = sorted(glob.glob(os.path.join(input_dir, "*.mrc")))
    elif os.path.isfile(input_dir):
        image_files = [input_dir]
    else:
        logger.error(f"crop.input_dir not found: {input_dir}")
        return False

    if not image_files:
        logger.warning(f"No .mrc files found in {input_dir}")
        return True

    logger.info(f"Cropping {len(image_files)} images from {input_dir}")
    t0 = time.time()

    for image_file in tqdm(image_files, desc="Cropping", file=sys.stdout):
        x_start, x_end, y_start, y_end = crop_average(
            image_file, output_averages, processing_dir,
            filter_size=c["filter_window"],
            mask_threshold=c["mask_threshold"],
            trim=c["trim"],
            crop_x=c["crop_x"],
            crop_y=c["crop_y"],
        )
        if crop_frames_flag:
            crop_frames_for_image(image_file, frames_dir, output_frames,
                                  x_start, x_end, y_start, y_end)

    logger.info(f"✓ crop completed in {time.time() - t0:.1f}s")
    return True


def run_blend(cfg: dict, logger: logging.Logger, dry_run: bool) -> bool:
    from .blend_tiles import discover_tilt_series, process_tilt_series

    c = cfg["blend"]

    out_avg      = os.path.join(c["output_dir"], "averages")
    out_frm      = os.path.join(c["output_dir"], "frames")
    out_avg_mdoc = os.path.join(out_avg, "mdocs")
    out_frm_mdoc = os.path.join(out_frm, "mdocs")
    proc_avg     = os.path.join(c["processing_dir"], "blending_averages")
    proc_frm     = os.path.join(c["processing_dir"], "blending_frames")

    if dry_run:
        logger.info("[DRY RUN] blend step:")
        for k, v in c.items():
            logger.info(f"  {k}: {v}")
        return True

    for d in [out_avg, out_avg_mdoc, proc_avg]:
        os.makedirs(d, exist_ok=True)
    if c["blend_frames"]:
        for d in [out_frm, out_frm_mdoc, proc_frm]:
            os.makedirs(d, exist_ok=True)

    ts_list = discover_tilt_series(c["mdoc_dir"])
    if not ts_list:
        logger.error(f"No tilt-series found in {c['mdoc_dir']}")
        return False

    ts_filter = c.get("ts_filter") or []
    if ts_filter:
        ts_list = [ts for ts in ts_list if ts in ts_filter]
        if not ts_list:
            logger.error("No tilt-series matched ts_filter")
            return False

    logger.info(f"Blending {len(ts_list)} tilt-series")
    t0 = time.time()

    for i, ts in enumerate(ts_list):
        logger.info(f"[{i + 1}/{len(ts_list)}] {ts}")
        process_tilt_series(
            ts=ts,
            mdoc_dir=c["mdoc_dir"],
            cropped_averages_dir=c["averages_dir"],
            cropped_frames_dir=c["frames_dir"],
            processing_averages_dir=proc_avg,
            processing_frames_dir=proc_frm,
            output_averages_dir=out_avg,
            output_frames_dir=out_frm,
            output_averages_mdoc_dir=out_avg_mdoc,
            output_frames_mdoc_dir=out_frm_mdoc,
            blend_size=c["blend_size"],
            blend_frames=c["blend_frames"],
            num_frames=c["num_frames"],
            normalize_averages_to_center=c.get("normalize_averages_to_center", False),
            normalize_frames_to_center=c.get("normalize_frames_to_center", False),
        )

    logger.info(f"✓ blend completed in {time.time() - t0:.1f}s")
    return True


def run_fill(cfg: dict, logger: logging.Logger, dry_run: bool) -> bool:
    from .remove_gaps import process_image, _process_with_gpu
    import concurrent.futures
    import glob
    import tqdm as tqdm_mod

    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required for the fill step. pip install torch")
        return False

    c = cfg["fill"]

    if dry_run:
        logger.info("[DRY RUN] fill step:")
        for k, v in c.items():
            logger.info(f"  {k}: {v}")
        return True

    os.makedirs(c["output_dir"], exist_ok=True)
    os.makedirs(c["mask_dir"],   exist_ok=True)

    image_list = sorted(glob.glob(os.path.join(c["input_dir"], "*.mrc")))
    if not image_list:
        logger.warning(f"No .mrc files found in {c['input_dir']}")
        return True

    logger.info(f"Filling gaps in {len(image_list)} images")
    t0    = time.time()
    gpus  = str(c["gpus"])
    sigma = float(c["sigma"])
    tile_num = int(c["tile_num"])
    resume   = bool(c["resume"])

    if gpus.lower() == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("Running gap-fill on CPU")
        for img in tqdm_mod.tqdm(image_list, desc="Filling gaps", file=sys.stdout):
            process_image(img, c["output_dir"], c["mask_dir"], device, resume, sigma, tile_num)
    else:
        available = torch.cuda.device_count()
        gpu_ids   = [int(g.strip()) for g in gpus.split(",") if g.strip().isdigit()]
        gpu_ids   = [g for g in gpu_ids if g < available] or [0]
        logger.info(f"Running gap-fill on GPUs: {gpu_ids}")

        task_args = [
            (img, gpu_ids[i % len(gpu_ids)], c["output_dir"], c["mask_dir"], resume, sigma, tile_num)
            for i, img in enumerate(image_list)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = [executor.submit(_process_with_gpu, a) for a in task_args]
            for future in tqdm_mod.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(image_list), desc="Filling gaps", file=sys.stdout,
            ):
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"Gap-fill error: {exc}")

    logger.info(f"✓ fill completed in {time.time() - t0:.1f}s")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STEP_RUNNERS = {
    "crop":  run_crop,
    "blend": run_blend,
    "fill":  run_fill,
}

STEP_NAMES = list(STEP_RUNNERS.keys())


@click.command()
@click.option("--config", "-c",
              type=click.Path(exists=True),
              help="Path to pipeline YAML config file.")
@click.option("--steps", "-s",
              multiple=True,
              type=click.Choice(STEP_NAMES, case_sensitive=False),
              help=(
                  "Steps to run. Repeatable: --steps crop --steps blend. "
                  "Overrides the steps.* flags in the config. "
                  "Defaults to whatever is enabled in the config."
              ))
@click.option("--dry-run", is_flag=True, default=False,
              help="Print what would be done without running anything.")
@click.option("--write-config",
              type=click.Path(),
              default=None,
              help="Write a template config YAML to this path and exit.")
def main(config, steps, dry_run, write_config):
    """Run the square-aperture montage pipeline from a YAML config file.

    \b
    Quickstart
    ----------
    1. Generate a template config:
         sam-run --write-config pipeline.yaml

    2. Edit pipeline.yaml to match your data layout.

    3. Run the full pipeline:
         sam-run --config pipeline.yaml

    4. Run only specific steps:
         sam-run --config pipeline.yaml --steps crop --steps blend
    """
    # ---- write-config mode ----
    if write_config:
        if yaml is None:
            print("ERROR: PyYAML is required. pip install pyyaml")
            sys.exit(1)
        out = Path(write_config)
        with open(out, "w") as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Template config written to: {out}")
        print(f"Edit it, then run:  sam-run --config {out}")
        return

    # ---- require config for all other modes ----
    if not config:
        click.echo("ERROR: --config is required (or use --write-config to generate a template).")
        sys.exit(1)

    # ---- load config ----
    cfg = load_config(config)

    # ---- set up logging ----
    log_cfg = cfg.get("logging", {})
    logger  = setup_logging(
        log_dir  = log_cfg.get("log_dir", "logs"),
        level_str= log_cfg.get("level",   "INFO"),
        dry_run  = dry_run,
    )

    # ---- change to data directory ----
    data_dir = cfg.get("data_dir", ".")
    if data_dir != ".":
        if not os.path.isdir(data_dir):
            logger.error(f"data_dir does not exist: {data_dir}")
            sys.exit(1)
        os.chdir(data_dir)
        logger.info(f"Working directory: {os.getcwd()}")

    # ---- determine which steps to run ----
    if steps:
        # CLI --steps flags override config
        enabled_steps = list(steps)
    else:
        steps_cfg     = cfg.get("steps", {})
        enabled_steps = [s for s in STEP_NAMES if steps_cfg.get(s, True)]

    if not enabled_steps:
        logger.warning("No steps enabled. Nothing to do.")
        return

    mode = "[DRY RUN] " if dry_run else ""
    logger.info(f"{mode}Pipeline steps: {' → '.join(enabled_steps)}")

    # ---- run each step ----
    pipeline_start = time.time()
    failed_steps   = []

    for step in enabled_steps:
        runner = STEP_RUNNERS[step]
        ok = runner(cfg, logger, dry_run)
        if not ok:
            failed_steps.append(step)
            logger.error(f"Step '{step}' failed — stopping pipeline.")
            break

    # ---- summary ----
    total = time.time() - pipeline_start
    logger.info("=" * 60)
    if failed_steps:
        logger.error(f"Pipeline FAILED at step '{failed_steps[0]}' after {total:.1f}s")
        sys.exit(1)
    else:
        logger.info(f"Pipeline completed successfully in {total:.1f}s")


if __name__ == "__main__":
    main()
