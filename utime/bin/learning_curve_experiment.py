"""
Learning curve experiment runner for U-Time fine-tuning.

Iterates over a configurable set of training set sizes N and, for each
intermediate size, independently draws k random subject-level subsamples from
the configured training pools.  Sampling is proportional to the number of
subjects available in each dataset's train pool so the dataset mixture
observed during full fine-tuning is preserved at every N.

Folder layout produced under --out_dir:

  N_004/
    rep_01/
      hyperparameters/          <- prototype hparams with train_data.data_dir patched
      model/                    <- symlink to pretrained weights file
      data_views/
        donders_2022/train/     <- symlinks to sampled subject folders
        wearanize_plus/train/   <- symlinks to sampled subject folders
    rep_02/
      ...
  N_008/
    rep_01/
      ...
  N_000/                        <- zero-shot: no training, prototype weights only
    rep_01/
      ...

After all runs complete a summary CSV is written to --out_dir/learning_curve_results.csv
by collecting the grand-mean Dice scores written by `ut cm --write_eval`.
"""

import logging
import os
import re
import json
import random
import subprocess
import argparse
import time
from collections import defaultdict
from glob import glob
from multiprocessing import Process, Lock, Queue, Event

from utime.bin.init import init_project_folder
from utime.hyperparameters import YAMLHParams
from utime.utils.scriptutils import add_logging_file_handler
from utime.utils.system import get_free_gpus, gpu_string_to_list

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Field name used in dataset config YAMLs to declare subject-grouping
# ------------------------------------------------------------------
_SUBJECT_REGEX_KEY = "subject_matching_regex"


# ======================================================================
# Argument parser
# ======================================================================

def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run a learning curve experiment by fine-tuning a pretrained U-Time "
            "model on systematically varied numbers of subjects.\n\n"
            "Reads dataset configs from --hparams_prototype_dir. "
            "Each dataset config YAML may contain an optional "
            f"'{_SUBJECT_REGEX_KEY}' field; if absent every file/folder in the "
            "train pool is treated as a separate subject."
        )
    )

    # ---- experiment design -------------------------------------------------
    parser.add_argument(
        "--training_sizes", type=int, nargs="+",
        default=[4, 8, 16, 24, 36, 48, 64, 96],
        help="Space-separated list of total training set sizes N to evaluate. "
             "Include 0 to add a zero-shot (direct-transfer) condition. "
             "Default: 4 8 16 24 36 48 64 96"
    )
    parser.add_argument(
        "--num_repeats", type=int, default=5,
        help="Number of independent random subject subsamples (repeats) per N "
             "for all N that are not the maximum training size. Default: 5"
    )
    parser.add_argument(
        "--full_size_repeats", type=int, default=1,
        help="Number of repeats for the maximum training size (no subsampling "
             "variance). Default: 1"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed used for reproducible subject subsampling and, "
             "unless --no_inject_seed, for injecting `ut --seed=<run_seed>` into "
             "each run's script commands. Default: 42"
    )
    parser.add_argument(
        "--no_inject_seed", action="store_true",
        help="Do not inject per-run `--seed` into `ut` script commands. "
             "Subject subsampling still uses the derived run seed. "
             "Use this to keep any `--seed` already present in the script "
             "prototype unchanged."
    )

    # ---- paths -------------------------------------------------------------
    parser.add_argument(
        "--out_dir", type=str, default="./lc_splits",
        help="Root output folder; one sub-folder per (N, rep) is created here. "
             "Default: ./lc_splits"
    )
    parser.add_argument(
        "--hparams_prototype_dir", type=str, default="./model_prototype",
        help="Prototype project directory whose hyperparameters/ sub-folder "
             "is copied into every run folder and whose model/ sub-folder "
             "provides the pretrained weights. Default: ./model_prototype"
    )
    parser.add_argument(
        "--pretrained_weights_dir", type=str, default=None,
        help="Optional explicit path to the directory containing the pretrained "
             "*.h5 weight file(s).  Defaults to <hparams_prototype_dir>/model/."
    )
    parser.add_argument(
        "--script_prototype", type=str, default="./script",
        help="Path to a text file listing `ut <command> ...` lines to execute "
             "inside each run folder.  GPU-related flags are stripped and "
             "--force_gpus=<assigned> is injected automatically. "
             "Unless --no_inject_seed, each run's derived seed is injected as "
             "`ut --seed=<run_seed>` (replacing any existing --seed). "
             "For the zero-shot condition (N=0) any line whose first token "
             "contains 'train' is automatically skipped. Default: ./script"
    )
    parser.add_argument(
        "--no_hparams", action="store_true",
        help="Do not copy prototype hyperparameters into each run folder "
             "(they must already be present)."
    )

    # ---- GPU / parallelism -------------------------------------------------
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to allocate to each parallel run. "
             "Also determines how many runs execute in parallel. Default: 1"
    )
    parser.add_argument(
        "--force_gpus", type=str, default="",
        help="Comma-separated GPU IDs to use (bypasses free-GPU detection)."
    )
    parser.add_argument(
        "--ignore_gpus", type=str, default="",
        help="Comma-separated GPU IDs to exclude from the pool."
    )
    parser.add_argument(
        "--num_jobs", type=int, default=1,
        help="Number of parallel jobs when --num_gpus=0. Default: 1"
    )
    parser.add_argument(
        "--monitor_gpus_every", type=int, default=None,
        help="If set, monitor GPU availability every N seconds and add "
             "newly freed GPUs to the resource pool."
    )

    # ---- run selection -----------------------------------------------------
    parser.add_argument(
        "--start_from", type=int, default=0,
        help="Start from this run index (0-based). Useful to resume a partial "
             "experiment. Default: 0"
    )
    parser.add_argument(
        "--run_on_index", type=int, default=None,
        help="Only execute a single run identified by its 0-based index in the "
             "ordered run list. Cannot be used together with --start_from."
    )
    parser.add_argument(
        "--wait_for", type=str, default="",
        help="PID (or comma-separated list) to wait for before starting."
    )

    # ---- logging -----------------------------------------------------------
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing log files."
    )
    parser.add_argument(
        "--log_file", type=str, default="lc_experiment.log",
        help="Log file name relative to the log directory. Default: lc_experiment_log"
    )

    return parser


# ======================================================================
# Subject / dataset helpers
# ======================================================================

def _pair_by_regex(paths, regex):
    """
    Group a list of file/folder paths by the first capture group of *regex*.

    Returns an ordered dict mapping subject_id -> [paths].
    Paths whose basename does not match the regex are treated as individual
    subjects (their full basename is used as the key).
    """
    pattern = re.compile(regex)
    groups = defaultdict(list)
    for p in sorted(paths):
        basename = os.path.basename(p)
        m = pattern.match(basename)
        if m:
            key = m.group(1)
        else:
            key = basename
        groups[key].append(p)
    return groups


def _get_subject_groups(train_data_dir, subject_matching_regex=None):
    """
    Return an ordered dict of subject_id -> [paths] for all entries
    (files **or** sub-folders) in *train_data_dir*.

    If *subject_matching_regex* is given, entries with matching subject IDs
    are merged into a single group (all their recordings are included together).
    Otherwise each entry is its own "subject".
    """
    entries = sorted(
        os.path.join(train_data_dir, e)
        for e in os.listdir(train_data_dir)
        if not e.startswith(".")
    )
    if not entries:
        raise ValueError(f"Train pool directory is empty: {train_data_dir}")

    if subject_matching_regex:
        return _pair_by_regex(entries, subject_matching_regex)
    else:
        return {os.path.basename(e): [e] for e in entries}


def _load_dataset_configs(hparams_prototype_dir):
    """
    Read the prototype hparams.yaml and return a dict:

        { dataset_id: {
            "train_data_dir":          str,
            "val_data_dir":            str,
            "test_data_dir":           str,
            "subject_matching_regex":  str | None,
            "yaml_rel_path":           str,   # relative path from hyperparameters/
          }, ...
        }
    """
    hparams_dir = os.path.join(hparams_prototype_dir, "hyperparameters")
    hparams_path = os.path.join(hparams_dir, "hparams.yaml")
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"Prototype hparams.yaml not found at {hparams_path}")

    top = YAMLHParams(hparams_path, no_version_control=True)
    if not top.get("datasets"):
        raise ValueError(
            f"Prototype hparams.yaml at {hparams_path} does not contain a "
            "'datasets' section. Only multi-dataset projects are supported."
        )

    configs = {}
    for ds_id, rel_path in top["datasets"].items():
        yaml_path = os.path.join(hparams_dir, rel_path)
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Dataset config for '{ds_id}' not found at {yaml_path}"
            )
        ds_hparams = YAMLHParams(yaml_path, no_version_control=True)
        configs[ds_id] = {
            "train_data_dir": ds_hparams.get("train_data", {}).get("data_dir"),
            "val_data_dir":   ds_hparams.get("val_data",   {}).get("data_dir"),
            "test_data_dir":  ds_hparams.get("test_data",  {}).get("data_dir"),
            "subject_matching_regex": ds_hparams.get(_SUBJECT_REGEX_KEY),
            "yaml_rel_path":  rel_path,
        }
    return configs


def _count_subjects(dataset_configs):
    """Return {dataset_id: n_subjects} from the actual train pools."""
    counts = {}
    for ds_id, cfg in dataset_configs.items():
        train_dir = cfg["train_data_dir"]
        if not train_dir or not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"Train data directory for '{ds_id}' does not exist: {train_dir}"
            )
        groups = _get_subject_groups(train_dir, cfg["subject_matching_regex"])
        counts[ds_id] = len(groups)
        logger.info(f"Dataset '{ds_id}': {counts[ds_id]} subjects in {train_dir}")
    return counts


def _compute_allocations(N, subject_counts):
    """
    Proportionally allocate N subjects across datasets.

    For each dataset d:
        n_d = round(N * pool_d / total_pool)
    with a correction to ensure sum == N (add/remove from the largest dataset).

    Returns {dataset_id: n_subjects}.
    """
    total = sum(subject_counts.values())
    if total == 0:
        return {ds: 0 for ds in subject_counts}
    ds_ids = list(subject_counts.keys())
    allocs = {ds: round(N * subject_counts[ds] / total) for ds in ds_ids}
    # Fix rounding drift
    diff = N - sum(allocs.values())
    if diff != 0:
        # Add/subtract from the dataset with the most subjects
        largest = max(ds_ids, key=lambda d: subject_counts[d])
        allocs[largest] += diff
    # Clamp to pool size
    for ds in ds_ids:
        allocs[ds] = max(0, min(allocs[ds], subject_counts[ds]))
    return allocs


# ======================================================================
# Run plan generation
# ======================================================================

def _build_run_plan(training_sizes, num_repeats, full_size_repeats, subject_counts, seed):
    """
    Return a list of dicts, one per run:

        {
            "N":   int,
            "rep": int,          # 1-based
            "allocations": {dataset_id: n_subjects},
            "seed": int,
        }

    Ordering: sorted by N (ascending) then rep (ascending).
    """
    sorted_sizes = sorted(training_sizes)
    max_N = max(s for s in sorted_sizes if s > 0) if any(s > 0 for s in sorted_sizes) else 0
    rng = random.Random(seed)

    plan = []
    for N in sorted_sizes:
        if N == 0:
            repeats = 1
        elif N == max_N:
            repeats = full_size_repeats
        else:
            repeats = num_repeats

        allocs = _compute_allocations(N, subject_counts)
        for rep in range(1, repeats + 1):
            run_seed = rng.randint(0, 2**31)
            plan.append({
                "N": N,
                "rep": rep,
                "allocations": allocs,
                "seed": run_seed,
            })
    return plan


def _run_dir_name(N, rep):
    return os.path.join(f"N_{N:03d}", f"rep_{rep:02d}")


# ======================================================================
# Data-view creation
# ======================================================================

def _sample_subjects(dataset_configs, allocations, run_seed):
    """
    For each dataset draw the allocated number of subjects.

    Returns {dataset_id: [list of entry paths to include in train view]}.
    """
    rng = random.Random(run_seed)
    sampled = {}
    for ds_id, n in allocations.items():
        if n == 0:
            sampled[ds_id] = []
            continue
        cfg = dataset_configs[ds_id]
        groups = _get_subject_groups(
            cfg["train_data_dir"],
            cfg["subject_matching_regex"]
        )
        subject_ids = list(groups.keys())
        chosen_ids = rng.sample(subject_ids, n)
        chosen_paths = [p for sid in chosen_ids for p in groups[sid]]
        sampled[ds_id] = chosen_paths
    return sampled


def _create_train_view(run_dir, dataset_id, paths):
    """
    Create <run_dir>/data_views/<dataset_id>/train/ and symlink each entry in
    *paths* into it.  Existing symlinks with the same name are overwritten.
    """
    view_dir = os.path.join(run_dir, "data_views", dataset_id, "train")
    os.makedirs(view_dir, exist_ok=True)
    for p in paths:
        dest = os.path.join(view_dir, os.path.basename(p))
        if os.path.lexists(dest):
            os.remove(dest)
        rel = os.path.relpath(p, view_dir)
        os.symlink(rel, dest)
    return view_dir


# ======================================================================
# Per-run project folder setup
# ======================================================================

def _patch_train_data_dir(run_dir, dataset_id, new_train_dir, yaml_rel_path):
    """
    Overwrite train_data.data_dir in the copied dataset config YAML.
    """
    yaml_path = os.path.join(run_dir, "hyperparameters", yaml_rel_path)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Dataset config not found in run dir: {yaml_path}")
    ds_hparams = YAMLHParams(yaml_path, no_version_control=True)
    ds_hparams.set_group(
        "/train_data/data_dir",
        new_train_dir,
        missing_parents_ok=True,
        overwrite=True,
    )
    ds_hparams.save_current(yaml_path)


def _link_pretrained_weights(run_dir, weights_dir):
    """
    Symlink the latest *.h5 file from *weights_dir* into <run_dir>/model/.

    If weights_dir is empty or has no .h5 files, a warning is logged and
    no symlink is created (training will start from scratch).
    """
    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    h5_files = sorted(glob(os.path.join(weights_dir, "*.h5")))
    if not h5_files:
        logger.warning(
            f"No *.h5 weight files found in {weights_dir}. "
            "Each run will train from a random initialisation."
        )
        return

    # Pick the latest by modification time
    latest = max(h5_files, key=os.path.getmtime)
    dest = os.path.join(model_dir, os.path.basename(latest))
    if os.path.lexists(dest):
        os.remove(dest)
    rel = os.path.relpath(latest, model_dir)
    os.symlink(rel, dest)
    logger.info(f"Linked pretrained weights: {latest} -> {dest}")


def _setup_run_folder(run_dir, run, dataset_configs, hparams_prototype_dir,
                      weights_dir, no_hparams):
    """
    Prepare a complete run folder:
      1. Copy prototype hparams (unless --no_hparams)
      2. For N > 0: create data views and patch train_data.data_dir
      3. Symlink pretrained weights
    """
    os.makedirs(run_dir, exist_ok=True)

    N = run["N"]
    allocations = run["allocations"]
    run_seed = run["seed"]

    # 1. Copy hyperparameters from prototype
    if not no_hparams:
        hparams_dir = os.path.join(hparams_prototype_dir, "hyperparameters")
        prototype_name = os.path.basename(hparams_dir)
        prototype_parent = os.path.dirname(hparams_dir)
        init_project_folder(
            default_folder=prototype_parent,
            preset=prototype_name,
            out_folder=run_dir,
            data_dir=None,   # we patch data_dirs manually below
        )

    # 2. Data views (only for N > 0)
    if N > 0:
        sampled = _sample_subjects(dataset_configs, allocations, run_seed)

        sampled_info = {
            ds: [os.path.basename(p) for p in paths]
            for ds, paths in sampled.items()
        }
        meta_path = os.path.join(run_dir, "sampled_subjects.json")
        with open(meta_path, "w") as f:
            json.dump({"N": N, "rep": run["rep"], "subjects": sampled_info}, f, indent=2)

        for ds_id, paths in sampled.items():
            cfg = dataset_configs[ds_id]
            view_dir = _create_train_view(run_dir, ds_id, paths)
            if not no_hparams:
                _patch_train_data_dir(
                    run_dir,
                    ds_id,
                    view_dir,
                    cfg["yaml_rel_path"],
                )

    # 3. Pretrained weights
    _link_pretrained_weights(run_dir, weights_dir)


# ======================================================================
# Command execution  (mirrors cv_experiment.py)
# ======================================================================

def _parse_script(script_path, gpus, skip_train=False):
    """
    Parse the script file and return a list of command-token lists.

    Lines are tokenised with shlex.split so that quoted arguments (e.g.
    glob patterns containing spaces or wildcards) are handled correctly —
    quotes are stripped and the contents are passed as a single token, just
    as a shell would do.

    GPU flags (and their space-separated values) are stripped and
    --force_gpus=<assigned> is injected for any command that originally
    declared a GPU flag.
    When *skip_train* is True, lines whose first token contains "train"
    are omitted (used for the zero-shot condition).
    """
    import shlex

    commands = []
    with open(script_path) as fh:
        for line in fh:
            line = line.strip(" \n")
            if not line or line.startswith("#"):
                continue
            line = line.split("#")[0].strip()
            if not line:
                continue
            try:
                tokens = shlex.split(line)
            except ValueError as exc:
                logger.warning(f"Could not parse script line (skipping): {line!r} — {exc}")
                continue
            if skip_train and any("train" in t.lower() for t in tokens[:2]):
                logger.info(f"[zero-shot] Skipping: {line}")
                continue
            # Strip GPU-related flags AND their space-separated values.
            # Flags using = (e.g. --force_gpus=0) are fully contained in one token.
            had_gpu_flag = any("gpu" in t.lower() for t in tokens)
            cmd = []
            skip_next = False
            for tok in tokens:
                if skip_next:
                    skip_next = False
                    continue
                if "gpu" in tok.lower():
                    # Flag without '=' is followed by a separate value token
                    if "=" not in tok:
                        skip_next = True
                    continue
                cmd.append(tok)
            # Re-inject the assigned GPU for any command that originally declared
            # a GPU flag (covers both `ut train/predict` and python/mp/ds launchers).
            if had_gpu_flag or "python" in line or line[:2] in ("mp", "ds"):
                cmd.append(f"--force_gpus={gpus}")
            commands.append(cmd)
    return commands


def _inject_ut_seed(cmd, seed):
    """
    Ensure a top-level ``ut --seed=<seed>`` is present on *cmd*.

    Only modifies commands whose first token is ``ut`` (or a path ending in
    ``ut``). Any existing ``--seed`` / ``--seed=...`` tokens (and a following
    value for the space-separated form) are removed before injecting the new
    flag immediately after ``ut``. Non-``ut`` commands are returned unchanged.
    """
    if not cmd:
        return cmd
    if os.path.basename(cmd[0]) != "ut":
        return cmd

    cleaned = []
    skip_next = False
    for tok in cmd[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok == "--seed":
            skip_next = True
            continue
        if tok.startswith("--seed="):
            continue
        cleaned.append(tok)
    return [cmd[0], f"--seed={int(seed)}"] + cleaned


def _run_sub_experiment(run_dir, script_path, gpus, gpu_queue, lock, N,
                        seed=None, inject_seed=True):
    run_label = os.path.relpath(run_dir)
    commands = _parse_script(script_path, gpus, skip_train=(N == 0))
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
        f"    N={N}  GPUs={gpus}{seed_info}\n"
        f"    Commands:\n" +
        "\n".join(f"      ({i+1}) {' '.join(c)}" for i, c in enumerate(commands)) +
        f"\n{sep}"
    )
    lock.release()

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
# GPU helpers  (copied from cv_experiment.py)
# ======================================================================

def _get_gpu_sets(free_gpus, num_gpus):
    free_gpus = list(map(str, free_gpus))
    return [",".join(free_gpus[x:x + num_gpus]) for x in range(0, len(free_gpus), num_gpus)]


def _get_free_gpu_sets(num_gpus, ignore_gpus=None):
    ignore_gpus = gpu_string_to_list(ignore_gpus or "", as_int=True)
    free_gpus = sorted(get_free_gpus())
    free_gpus = [g for g in free_gpus if g not in ignore_gpus]
    total = len(free_gpus)
    if total < num_gpus or total % num_gpus:
        if total < num_gpus:
            raise ValueError(
                f"Requested {num_gpus} GPUs per run but only {total} are free."
            )
        raise NotImplementedError("Uneven GPU distribution not supported.")
    return _get_gpu_sets(free_gpus, num_gpus)


def _monitor_gpus(every, gpu_queue, num_gpus, ignore_gpus, current_pool, stop_event):
    current_pool = [g for subset in current_pool for g in subset.split(",")]
    while not stop_event.is_set():
        try:
            for gpu_set in _get_free_gpu_sets(num_gpus, ignore_gpus):
                if any(g in current_pool for g in gpu_set.split(",")):
                    continue
                gpu_queue.put(gpu_set)
                current_pool += gpu_set.split(",")
        except (ValueError, NotImplementedError):
            pass
        finally:
            time.sleep(every)


def _start_gpu_monitor(args, gpu_queue, gpu_sets):
    procs, stop_event = [], None
    if args.monitor_gpus_every:
        stop_event = Event()
        t = Process(
            target=_monitor_gpus,
            args=(args.monitor_gpus_every, gpu_queue, args.num_gpus,
                  args.ignore_gpus, gpu_sets, stop_event)
        )
        t.start()
        procs.append(t)
    return procs, stop_event


# ======================================================================
# Result aggregation
# ======================================================================

def _collect_results(out_dir, run_plan):
    """
    Walk completed run folders and aggregate overall metrics written by `ut cm`
    (`overall_metrics.json`) into a single summary CSV.
    """
    import pandas as pd

    def _read_overall_metrics_json(path):
        import json
        import numpy as np
        try:
            with open(path, "r") as f:
                d = json.load(f)
            def _get(key):
                v = d.get(key, np.nan)
                try:
                    return float(v)
                except Exception:
                    return np.nan
            return {
                "accuracy": _get("accuracy"),
                "macro_f1": _get("macro_f1"),
                "micro_f1": _get("micro_f1"),
                "kappa": _get("kappa"),
            }
        except Exception:
            return {"accuracy": np.nan, "macro_f1": np.nan, "micro_f1": np.nan, "kappa": np.nan}

    rows = []
    for run in run_plan:
        run_dir = os.path.join(out_dir, _run_dir_name(run["N"], run["rep"]))
        # `ut cm` writes overall_metrics.json to its output dir (often `evaluations/`)
        for metrics_json in glob(os.path.join(run_dir, "**", "overall_metrics.json"), recursive=True):
            try:
                overall = _read_overall_metrics_json(metrics_json)
                # Infer dataset from metrics path (first folder under run_dir)
                rel = os.path.relpath(metrics_json, run_dir)
                rows.append({
                    "N": run["N"],
                    "rep": run["rep"],
                    "dataset": rel.split(os.sep)[0] if os.sep in rel else "combined",
                    "overall_metrics_json": metrics_json,
                    "accuracy": overall["accuracy"],
                    "macro_f1": overall["macro_f1"],
                    "micro_f1": overall["micro_f1"],
                    "kappa": overall["kappa"],
                    "seed": run["seed"],
                })
            except Exception as e:
                logger.warning(f"Could not read {metrics_json}: {e}")

    if not rows:
        logger.warning("No overall_metrics.json files found; summary CSV not written.")
        return

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "learning_curve_results.csv")
    summary.to_csv(out_csv, index=False)
    logger.info(f"Learning curve results written to {out_csv}")

    # Also print per-N mean ± std
    try:
        metrics = ["accuracy", "macro_f1", "micro_f1", "kappa"]
        avail = [m for m in metrics if m in summary.columns]
        if not avail:
            raise KeyError(f"No metric columns found in summary. Expected one of {metrics}")

        grouped = summary.groupby(["N", "dataset"])
        parts = []
        for m in avail:
            a = grouped[m].agg(["mean", "std", "count"]).reset_index()
            # Rename stat cols to be metric-specific, keep grouping keys
            a = a.rename(columns={
                "mean": f"{m}_mean",
                "std": f"{m}_std",
                "count": f"{m}_count",
            })
            parts.append(a)

        # Merge per-metric summaries on (N, dataset)
        agg = parts[0]
        for a in parts[1:]:
            agg = agg.merge(a, on=["N", "dataset"], how="outer")
        logger.info(f"\nLearning curve summary:\n{agg.to_string(index=False)}")
    except Exception as e:
        logger.warning(f"Could not aggregate results: {e}")


# ======================================================================
# Main run
# ======================================================================

def run(args):
    hparams_prototype_dir = os.path.abspath(args.hparams_prototype_dir)
    out_dir = os.path.abspath(args.out_dir)
    script_path = os.path.abspath(args.script_prototype)
    weights_dir = (
        os.path.abspath(args.pretrained_weights_dir)
        if args.pretrained_weights_dir
        else os.path.join(hparams_prototype_dir, "model")
    )

    # Validate inputs
    if not os.path.isdir(hparams_prototype_dir):
        raise FileNotFoundError(
            f"Prototype directory not found: {hparams_prototype_dir}"
        )
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Script prototype not found: {script_path}")
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"Pretrained weights directory not found: {weights_dir}"
        )

    # Assert GPU flags are compatible
    force_gpus = gpu_string_to_list(args.force_gpus)
    ignore_gpus = gpu_string_to_list(args.ignore_gpus)
    overlap = set(force_gpus) & set(ignore_gpus)
    if overlap:
        raise RuntimeError(f"Cannot both force and ignore GPU(s) {overlap}.")
    if args.run_on_index is not None and args.start_from:
        raise RuntimeError("Cannot use both --run_on_index and --start_from.")

    # Load dataset configs and discover subject pools
    dataset_configs = _load_dataset_configs(hparams_prototype_dir)
    subject_counts = _count_subjects(dataset_configs)

    # Build the ordered run plan
    run_plan = _build_run_plan(
        training_sizes=args.training_sizes,
        num_repeats=args.num_repeats,
        full_size_repeats=args.full_size_repeats,
        subject_counts=subject_counts,
        seed=args.seed,
    )

    logger.info(
        f"\nRun plan ({len(run_plan)} total runs):\n" +
        "\n".join(
            f"  [{i:03d}] N={r['N']:3d}  rep={r['rep']:2d}  "
            f"allocs={r['allocations']}  seed={r['seed']}"
            for i, r in enumerate(run_plan)
        )
    )

    # Wait for external PID if requested
    if args.wait_for:
        from utime.utils import await_pids
        await_pids(args.wait_for)

    # Optionally restrict to a single run
    if args.run_on_index is not None:
        if args.run_on_index < 0 or args.run_on_index >= len(run_plan):
            raise RuntimeError(
                f"--run_on_index {args.run_on_index} out of range "
                f"[0, {len(run_plan)-1}]."
            )
        run_plan = [run_plan[args.run_on_index]]
        start = 0
    else:
        start = args.start_from

    # Determine GPU / job slots
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

    # Pre-create all run folders (sampling is deterministic; safe to do eagerly)
    os.makedirs(out_dir, exist_ok=True)
    for run_info in run_plan:
        run_dir = os.path.join(out_dir, _run_dir_name(run_info["N"], run_info["rep"]))
        logger.info(f"Setting up run folder: {run_dir}")
        _setup_run_folder(
            run_dir=run_dir,
            run=run_info,
            dataset_configs=dataset_configs,
            hparams_prototype_dir=hparams_prototype_dir,
            weights_dir=weights_dir,
            no_hparams=args.no_hparams,
        )

    # GPU monitor
    running_procs, stop_event = _start_gpu_monitor(args, gpu_queue, gpu_sets)

    try:
        for run_info in run_plan[start:]:
            gpus = gpu_queue.get()
            run_dir = os.path.join(out_dir, _run_dir_name(run_info["N"], run_info["rep"]))
            t = Process(
                target=_run_sub_experiment,
                args=(
                    run_dir,
                    script_path,
                    gpus,
                    gpu_queue,
                    lock,
                    run_info["N"],
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
        logger.warning("Interrupted – terminating child processes.")
        for p in running_procs:
            p.terminate()

    if stop_event is not None:
        stop_event.set()
    for p in running_procs:
        p.join()

    # Aggregate results across completed runs
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
