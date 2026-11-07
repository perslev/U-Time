"""
Grid search experiment runner for U-Time hyperparameter sweeps.

Iterates over a Cartesian product of hyperparameter values, creates one
project folder per combination (plus optional repeats), patches the specified
values into the copied hparams.yaml, and runs the configured pipeline.

Typical use — depth analysis:

    ut grid_search_experiment \\
      --hparams_prototype_dir /path/to/project \\
      --out_dir ./depth_variants \\
      --grid "build.depth=12,9,6,3" \\
      --script_prototype ./script \\
      --num_gpus 1

Multi-parameter grid (Cartesian product of all --grid specs):

    ut grid_search_experiment \\
      --grid "build.depth=6,9,12" \\
      --grid "fit.optimizer_kwargs.learning_rate=1e-4,1e-5" \\
      ...

Folder layout under --out_dir:

    depth_12/
      rep_01/
        hyperparameters/   <- prototype hparams with build.depth patched to 12
        model/             <- symlink to pretrained weights (if --pretrained_weights_dir given)
    depth_9/
      rep_01/
        ...

Training from scratch (no --pretrained_weights_dir) is the default.

Unless --no_inject_seed is set, each (combo, rep) run injects its derived
seed as `ut --seed=<run_seed>` into every `ut ...` line of the script
prototype (replacing any existing --seed on those lines).
"""

import logging
import os
 
import argparse
from itertools import product
from multiprocessing import Process, Lock, Queue


from utime.bin.init import init_project_folder
from utime.hyperparameters import YAMLHParams
from utime.utils.scriptutils import add_logging_file_handler

# Reuse GPU helpers and script parser from the learning curve script
from utime.bin.learning_curve_experiment import (
    _parse_script,
    _inject_ut_seed,
    _get_free_gpu_sets,
    _start_gpu_monitor,
    _link_pretrained_weights,
)
from utime.utils.system import gpu_string_to_list

logger = logging.getLogger(__name__)


# ======================================================================
# Argument parser
# ======================================================================

def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run a grid search experiment over U-Time hyperparameters. "
            "For each combination of values specified via --grid, a separate "
            "run folder is created with a patched copy of the prototype hparams."
        )
    )

    # ---- grid definition ---------------------------------------------------
    parser.add_argument(
        "--grid", type=str, action="append", dest="grid", default=None,
        metavar="KEY=V1,V2,...",
        help="Hyperparameter sweep specification. KEY is a dot-separated path "
             "into hparams.yaml (e.g. 'build.depth') and V1,V2,... are the "
             "values to sweep over. Repeat --grid for a multi-dimensional "
             "sweep (Cartesian product). Example: --grid 'build.depth=3,6,9,12'"
    )
    parser.add_argument(
        "--grid_file", type=str, default=None,
        metavar="PATH",
        help="Optional path to a YAML or JSON file specifying the grid as a "
             "mapping of dotted-key to list-of-values. Values in this file are "
             "merged with any --grid arguments."
    )
    parser.add_argument(
        "--num_repeats", type=int, default=1,
        help="Number of independent repeats per hyperparameter combination "
             "(useful when training from scratch with different random seeds). "
             "Each repeat gets a distinct derived seed injected into `ut` "
             "commands unless --no_inject_seed. Default: 1"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed. Each (combo, rep) derives its seed for "
             "deterministic reproducibility; that seed is injected as "
             "`ut --seed=<run_seed>` unless --no_inject_seed. Default: 42"
    )
    parser.add_argument(
        "--no_inject_seed", action="store_true",
        help="Do not inject per-run `--seed` into `ut` script commands."
    )

    # ---- paths -------------------------------------------------------------
    parser.add_argument(
        "--out_dir", type=str, default="./gs_splits",
        help="Root output folder; one sub-folder per (combo, rep) is created "
             "here. Default: ./gs_splits"
    )
    parser.add_argument(
        "--hparams_prototype_dir", type=str, default="./model_prototype",
        help="Prototype project directory whose hyperparameters/ sub-folder "
             "is copied into every run folder. Default: ./model_prototype"
    )
    parser.add_argument(
        "--pretrained_weights_dir", type=str, default=None,
        help="Optional path to a directory containing pretrained *.h5 weight "
             "file(s) to symlink into each run's model/ folder. "
             "If not given, runs train from scratch (model/ is left empty)."
    )
    parser.add_argument(
        "--script_prototype", type=str, default="./script",
        help="Path to the script file listing `ut <command> ...` lines to "
             "execute inside each run folder. Default: ./script"
    )
    parser.add_argument(
        "--no_hparams", action="store_true",
        help="Do not copy prototype hyperparameters into each run folder."
    )

    # ---- GPU / parallelism -------------------------------------------------
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="GPUs per run; also determines how many runs execute in parallel."
    )
    parser.add_argument("--force_gpus", type=str, default="")
    parser.add_argument("--ignore_gpus", type=str, default="")
    parser.add_argument(
        "--num_jobs", type=int, default=1,
        help="Parallel jobs when --num_gpus=0. Default: 1"
    )
    parser.add_argument("--monitor_gpus_every", type=int, default=None)

    # ---- run selection -----------------------------------------------------
    parser.add_argument(
        "--start_from", type=int, default=0,
        help="Start from this run index (0-based). Default: 0"
    )
    parser.add_argument(
        "--run_on_index", type=int, default=None,
        help="Only execute the single run at this 0-based index."
    )
    parser.add_argument("--wait_for", type=str, default="")

    # ---- logging -----------------------------------------------------------
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log_file", type=str, default="gs_experiment_log",
        help="Log file name relative to the log directory. Default: gs_experiment_log"
    )

    return parser


# ======================================================================
# Grid parsing
# ======================================================================

def _coerce_value(s):
    """Try to parse a string as int, then float, then bool, else keep as str."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_grid_specs(grid_args, grid_file):
    """
    Parse --grid KEY=V1,V2,... arguments and an optional --grid_file into an
    ordered list of (dotted_key, [values]) pairs.

    Ordering is preserved so that folder names are deterministic.
    """
    specs = {}  # dotted_key -> [values]

    if grid_file:
        import json
        import yaml as _yaml
        with open(grid_file) as fh:
            content = fh.read()
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            raw = _yaml.safe_load(content)
        for key, vals in raw.items():
            specs[key] = [_coerce_value(str(v)) for v in (vals if isinstance(vals, list) else [vals])]

    for spec in (grid_args or []):
        if "=" not in spec:
            raise ValueError(
                f"Invalid --grid specification '{spec}'. "
                "Expected format: KEY=V1,V2,..."
            )
        key, rest = spec.split("=", 1)
        key = key.strip()
        values = [_coerce_value(v.strip()) for v in rest.split(",")]
        specs[key] = values

    if not specs:
        raise ValueError(
            "No grid specifications found. "
            "Use --grid 'key=v1,v2' or --grid_file."
        )
    return list(specs.items())  # [(dotted_key, [values]), ...]


def _dot_key_to_path(dotted_key):
    """Convert 'build.depth' -> '/build/depth' for YAMLHParams.set_group."""
    return "/" + dotted_key.replace(".", "/")


def _value_for_folder(v):
    """Format a value for use in a directory name."""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _combo_to_dir_name(combo):
    """
    Build a directory name from a list of (dotted_key, value) pairs.

    'build.depth=12' -> 'depth_12'
    Multiple: 'depth_12_learning_rate_1e-05'
    """
    parts = []
    for dotted_key, value in combo:
        short_key = dotted_key.split(".")[-1]
        parts.append(f"{short_key}_{_value_for_folder(value)}")
    return "_".join(parts)


# ======================================================================
# Run plan
# ======================================================================

def _build_run_plan(grid_specs, num_repeats, seed):
    """
    Return a list of run dicts:

        {
          "combo":      [(dotted_key, value), ...],   # the hparam values for this run
          "combo_name": str,                           # directory-safe name
          "rep":        int,                           # 1-based
          "seed":       int,
        }

    Ordered: Cartesian product in grid_specs order, then rep ascending.
    """
    import random
    rng = random.Random(seed)
    keys = [k for k, _ in grid_specs]
    value_lists = [vs for _, vs in grid_specs]

    plan = []
    for values in product(*value_lists):
        combo = list(zip(keys, values))
        combo_name = _combo_to_dir_name(combo)
        for rep in range(1, num_repeats + 1):
            run_seed = rng.randint(0, 2**31)
            plan.append({
                "combo": combo,
                "combo_name": combo_name,
                "rep": rep,
                "seed": run_seed,
            })
    return plan


def _run_dir_name(combo_name, rep):
    return os.path.join(combo_name, f"rep_{rep:02d}")


# ======================================================================
# Per-run setup
# ======================================================================

def _patch_hparams(run_dir, combo):
    """
    Apply the (dotted_key, value) pairs in *combo* to the copied hparams.yaml.
    """
    hparams_path = os.path.join(run_dir, "hyperparameters", "hparams.yaml")
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(
            f"hparams.yaml not found in run dir: {hparams_path}"
        )
    hparams = YAMLHParams(hparams_path, no_version_control=True)
    for dotted_key, value in combo:
        path = _dot_key_to_path(dotted_key)
        logger.info(f"  Patching {path} = {value!r}")
        hparams.set_group(path, value, overwrite=True, missing_parents_ok=True)
    hparams.save_current(hparams_path)


def _setup_run_folder(run_dir, run, hparams_prototype_dir, pretrained_weights_dir, no_hparams):
    """
    Prepare a complete run folder:
      1. Copy prototype hparams (unless --no_hparams)
      2. Patch the specified hyperparameter values
      3. Optionally symlink pretrained weights
    """
    os.makedirs(run_dir, exist_ok=True)

    if not no_hparams:
        hparams_dir = os.path.join(hparams_prototype_dir, "hyperparameters")
        init_project_folder(
            default_folder=os.path.dirname(hparams_dir),
            preset=os.path.basename(hparams_dir),
            out_folder=run_dir,
            data_dir=None,
        )
        _patch_hparams(run_dir, run["combo"])

    if pretrained_weights_dir:
        _link_pretrained_weights(run_dir, pretrained_weights_dir)


# ======================================================================
# Command execution
# ======================================================================

def _run_sub_experiment(run_dir, script_path, gpus, gpu_queue, lock,
                        seed=None, inject_seed=True):
    run_label = os.path.relpath(run_dir)
    commands = _parse_script(script_path, gpus, skip_train=False)
    if inject_seed and seed is not None:
        commands = [_inject_ut_seed(c, seed) for c in commands]

    orig_dir = os.getcwd()
    os.chdir(run_dir)

    lock.acquire()
    sep = "-" * 60
    seed_info = f"  seed={seed}" if seed is not None else ""
    logger.info(
        f"\n{sep}\n"
        f"[*] Run: {run_label}\n"
        f"    GPUs={gpus}{seed_info}\n"
        f"    Commands:\n" +
        "\n".join(f"      ({i+1}) {' '.join(c)}" for i, c in enumerate(commands)) +
        f"\n{sep}"
    )
    lock.release()

    import subprocess
    run_next = True
    for cmd in commands:
        if not run_next:
            break
        str_cmd = " ".join(cmd)
        lock.acquire()
        logger.info(f"[{run_label} - STARTING] {str_cmd}")
        lock.release()

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = p.communicate()
        rc = p.returncode

        lock.acquire()
        if rc != 0:
            logger.error(
                f"[{run_label} - ERROR {rc}] {str_cmd}\n"
                f"----- stderr -----\n{err.decode('utf-8')}\n"
                f"----- end -----\n"
            )
            run_next = False
        else:
            logger.info(f"[{run_label} - DONE] {str_cmd}")
        lock.release()

    os.chdir(orig_dir)
    gpu_queue.put(gpus)


# ======================================================================
# Result aggregation
# ======================================================================

def _collect_results(out_dir, run_plan):
    """
    Walk completed run folders and aggregate overall metrics written by `ut cm`
    (`overall_metrics.json`) into a single summary CSV.
    """
    import pandas as pd
    import numpy as np
    from glob import glob

    def _read_overall_metrics_json(path):
        import json
        try:
            with open(path, "r") as f:
                d = json.load(f)
            def _get(key):
                v = d.get(key, np.nan)
                try:
                    return float(v)
                except Exception:
                    return np.nan
            return {"accuracy": _get("accuracy"), "macro_f1": _get("macro_f1"), "micro_f1": _get("micro_f1"), "kappa": _get("kappa")}
        except Exception:
            return {"accuracy": np.nan, "macro_f1": np.nan, "micro_f1": np.nan, "kappa": np.nan}

    rows = []
    for run in run_plan:
        run_dir = os.path.join(out_dir, _run_dir_name(run["combo_name"], run["rep"]))
        for metrics_json in glob(os.path.join(run_dir, "**", "overall_metrics.json"), recursive=True):
            try:
                overall = _read_overall_metrics_json(metrics_json)
                rel = os.path.relpath(metrics_json, run_dir)
                row = {
                    "combo_name": run["combo_name"],
                    "rep": run["rep"],
                    "seed": run["seed"],
                    "dataset": rel.split(os.sep)[0] if os.sep in rel else "combined",
                    "overall_metrics_json": metrics_json,
                    "accuracy": overall["accuracy"],
                    "macro_f1": overall["macro_f1"],
                    "micro_f1": overall["micro_f1"],
                    "kappa": overall["kappa"],
                }
                for dotted_key, value in run["combo"]:
                    row[dotted_key] = value
                rows.append(row)
            except Exception as e:
                logger.warning(f"Could not read {metrics_json}: {e}")

    if not rows:
        logger.warning("No overall_metrics.json files found; summary CSV not written.")
        return

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "grid_search_results.csv")
    summary.to_csv(out_csv, index=False)
    logger.info(f"Grid search results written to {out_csv}")

    try:
        hparam_keys = [k for k, _ in run_plan[0]["combo"]] if run_plan else []
        metrics = ["accuracy", "macro_f1", "micro_f1", "kappa"]
        avail = [m for m in metrics if m in summary.columns]
        if not avail:
            raise KeyError(f"No metric columns found in summary. Expected one of {metrics}")

        grouped = summary.groupby(hparam_keys + ["dataset"])
        parts = []
        for m in avail:
            a = grouped[m].agg(["mean", "std", "count"]).reset_index()
            a = a.rename(columns={
                "mean": f"{m}_mean",
                "std": f"{m}_std",
                "count": f"{m}_count",
            })
            parts.append(a)

        agg = parts[0]
        for a in parts[1:]:
            agg = agg.merge(a, on=hparam_keys + ["dataset"], how="outer")
        logger.info(f"\nGrid search summary:\n{agg.to_string(index=False)}")
    except Exception as e:
        logger.warning(f"Could not aggregate results: {e}")


# ======================================================================
# Main run
# ======================================================================

def run(args):
    hparams_prototype_dir = os.path.abspath(args.hparams_prototype_dir)
    out_dir = os.path.abspath(args.out_dir)
    script_path = os.path.abspath(args.script_prototype)
    pretrained_weights_dir = (
        os.path.abspath(args.pretrained_weights_dir)
        if args.pretrained_weights_dir
        else None
    )

    if not os.path.isdir(hparams_prototype_dir):
        raise FileNotFoundError(f"Prototype directory not found: {hparams_prototype_dir}")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Script prototype not found: {script_path}")
    if pretrained_weights_dir and not os.path.isdir(pretrained_weights_dir):
        raise FileNotFoundError(f"Pretrained weights directory not found: {pretrained_weights_dir}")

    force_gpus = gpu_string_to_list(args.force_gpus)
    ignore_gpus = gpu_string_to_list(args.ignore_gpus)
    if set(force_gpus) & set(ignore_gpus):
        raise RuntimeError(f"Cannot both force and ignore GPU(s) {set(force_gpus) & set(ignore_gpus)}.")
    if args.run_on_index is not None and args.start_from:
        raise RuntimeError("Cannot use both --run_on_index and --start_from.")

    # Parse grid and build run plan
    grid_specs = _parse_grid_specs(args.grid, args.grid_file)
    run_plan = _build_run_plan(grid_specs, args.num_repeats, args.seed)

    logger.info(
        f"\nGrid search plan ({len(run_plan)} total runs):\n" +
        "\n".join(
            f"  [{i:03d}] {r['combo_name']}  rep={r['rep']:2d}  seed={r['seed']}"
            for i, r in enumerate(run_plan)
        )
    )

    if args.wait_for:
        from utime.utils import await_pids
        await_pids(args.wait_for)

    if args.run_on_index is not None:
        if not (0 <= args.run_on_index < len(run_plan)):
            raise RuntimeError(
                f"--run_on_index {args.run_on_index} out of range [0, {len(run_plan)-1}]."
            )
        run_plan = [run_plan[args.run_on_index]]
        start = 0
    else:
        start = args.start_from

    # GPU / job slots
    if args.force_gpus:
        from utime.utils import set_gpu
        set_gpu(args.force_gpus)
    if args.num_gpus:
        gpu_sets = _get_free_gpu_sets(args.num_gpus, args.ignore_gpus)[: len(run_plan)]
    elif args.num_jobs and args.num_jobs > 0:
        gpu_sets = ["''"] * args.num_jobs
    else:
        raise ValueError("Specify --num_gpus or --num_jobs > 0.")

    lock = Lock()
    gpu_queue = Queue()
    for g in gpu_sets:
        gpu_queue.put(g)

    # Pre-create all run folders
    os.makedirs(out_dir, exist_ok=True)
    for run_info in run_plan:
        run_dir = os.path.join(out_dir, _run_dir_name(run_info["combo_name"], run_info["rep"]))
        logger.info(f"Setting up run folder: {run_dir}  combo={run_info['combo']}")
        _setup_run_folder(
            run_dir=run_dir,
            run=run_info,
            hparams_prototype_dir=hparams_prototype_dir,
            pretrained_weights_dir=pretrained_weights_dir,
            no_hparams=args.no_hparams,
        )

    # GPU monitor
    running_procs, stop_event = _start_gpu_monitor(args, gpu_queue, gpu_sets)

    try:
        for run_info in run_plan[start:]:
            gpus = gpu_queue.get()
            run_dir = os.path.join(out_dir, _run_dir_name(run_info["combo_name"], run_info["rep"]))
            t = Process(
                target=_run_sub_experiment,
                args=(
                    run_dir,
                    script_path,
                    gpus,
                    gpu_queue,
                    lock,
                    run_info["seed"],
                    not args.no_inject_seed,
                ),
            )
            t.start()
            running_procs.append(t)
            for p in running_procs:
                if not p.is_alive():
                    p.join()
    except KeyboardInterrupt:
        logger.warning("Interrupted — terminating child processes.")
        for p in running_procs:
            p.terminate()

    if stop_event is not None:
        stop_event.set()
    for p in running_procs:
        p.join()

    _collect_results(out_dir, run_plan)


# ======================================================================
# Entry point
# ======================================================================

def entry_func(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="a")
    run(args)


if __name__ == "__main__":
    entry_func()
