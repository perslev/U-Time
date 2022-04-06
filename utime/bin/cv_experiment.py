import logging
import os
import time
import shutil
import argparse
import subprocess
from multiprocessing import Process, Lock, Queue, Event
from utime.utils import create_folders
from utime.bin.init import init_project_folder
from utime.utils.system import get_free_gpus, gpu_string_to_list
from utime.hyperparameters import YAMLHParams
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description=
        "Run a CV experiment. First, split a dataset folder using the 'ut cv_split' command. " + \
        "This command may then be used to invoke a set of specified commands (usually 'ut <command>' commands) on " + \
        "each split. By default the commands that should be run on each split must be specified in a file named 'script' (otherwise " + \
        "specify a different path via the --script_prototype flag). The current working directory must be a project folder as created by " + \
        "'ut init' or contain a subfolder at path '--hparams_prototype_dir <path>' storing the hyperparameters to use.")
    parser.add_argument("--cv_dir", type=str, required=True,
                        help="Directory storing split subfolders as output by"
                             " cv_split.py")
    parser.add_argument("--out_dir", type=str, default="./splits",
                        help="Folder in which experiments will be run and "
                             "results stored.")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use per process. This also "
                             "defines the number of parallel jobs to run.")
    parser.add_argument("--force_gpus", type=str, default="",
                        help="A list of one or more GPU IDs "
                             "(comma separated) from which GPU resources "
                             "will supplied to each split, independent of"
                             " the current memory usage of the GPUs.")
    parser.add_argument("--ignore_gpus", type=str, default="",
                        help="A list of one or more GPU IDs "
                             "(comma separated) that will not be considered.")
    parser.add_argument("--num_jobs", type=int, default=1,
                        help="OBS: Only in effect when --num_gpus=0. Sets"
                             " the number of jobs to run in parallel when no"
                             " GPUs are attached to each job.")
    parser.add_argument("--run_on_split", type=int, default=None,
                        help="Only run a specific split")
    parser.add_argument("--script_prototype", type=str, default="./script",
                        help="Path to text file listing commands and "
                             "arguments to execute under each sub-exp folder.")
    parser.add_argument("--hparams_prototype_dir", type=str,
                        default="./model_prototype",
                        help="Prototype directory storing all hyperparameter "
                             "yaml files from which sub-CV models will be run")
    parser.add_argument("--no_hparams", action="store_true",
                        help="Do not move a hyperparameter yaml file into "
                             "each split dir (one must be already there).")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Start from CV split<start_from>. Default 0.")
    parser.add_argument("--wait_for", type=str, default="",
                        help="Waiting for pid to terminate before starting "
                             "training process.")
    parser.add_argument("--monitor_gpus_every", type=int, default=None,
                        help="If specified, start a background process which"
                             " monitors every 'monitor_gpus_every' seconds "
                             "whether new GPUs have become available than may"
                             " be included in the CV experiment GPU resource "
                             "pool.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing log files.')
    parser.add_argument("--log_file", type=str, default="cv_experiment_log",
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is 'cv_experiment_log'")
    return parser


def get_cv_folders(dir_):
    key = lambda x: int(x.split("_")[-1])
    return [os.path.join(dir_, p) for p in sorted(os.listdir(dir_), key=key)]


def _get_gpu_sets(free_gpus, num_gpus):
    free_gpus = list(map(str, free_gpus))
    return [",".join(free_gpus[x:x + num_gpus]) for x in range(0, len(free_gpus), num_gpus)]


def get_free_gpu_sets(num_gpus, ignore_gpus=None):
    ignore_gpus = gpu_string_to_list(ignore_gpus or "", as_int=True)
    free_gpus = sorted(get_free_gpus())
    free_gpus = list(filter(lambda gpu: gpu not in ignore_gpus, free_gpus))
    total_gpus = len(free_gpus)
    if total_gpus % num_gpus or not free_gpus:
        if total_gpus < num_gpus:
            raise ValueError(f"Invalid number of GPUs per process '{num_gpus}' for total "
                             f"GPU count of '{total_gpus}' - must be evenly divisible.")
        else:
            raise NotImplementedError
    else:
        return _get_gpu_sets(free_gpus, num_gpus)


def monitor_gpus(every, gpu_queue, num_gpus, ignore_gpus, current_pool, stop_event):
    # Make flat version of the list of gpu sets
    current_pool = [gpu for sublist in current_pool for gpu in sublist.split(",")]
    while not stop_event.is_set():
        # Get available GPU sets. Will raise ValueError if no full set is
        # available
        try:
            gpu_sets = get_free_gpu_sets(num_gpus, ignore_gpus)
            for gpu_set in gpu_sets:
                if any([g in current_pool for g in gpu_set.split(",")]):
                    # If one or more GPUs are already in use - this may happen
                    # initially as preprocessing occurs in a process before GPU
                    # memory has been allocated - ignore the set
                    continue
                else:
                    gpu_queue.put(gpu_set)
                    current_pool += gpu_set.split(",")
        except ValueError:
            pass
        finally:
            time.sleep(every)


def parse_script(script, gpus):
    commands = []
    with open(script) as in_file:
        for line in in_file:
            line = line.strip(" \n")
            if not line or line[0] == "#":
                continue
            # Split out in-line comments
            line = line.split("#")[0]
            # Get all arguments, remove if concerning GPU (controlled here)
            cmd = list(filter(lambda x: "gpu" not in x.lower(), line.split()))
            if "python" in line or line[:2] == "mp" or line[:2] == "ds":
                cmd.append(f"--force_gpus={gpus}")
            commands.append(cmd)
    return commands


def run_sub_experiment(split_dir, out_dir, script, hparams_dir, no_hparams, gpus, gpu_queue, lock):
    # Create sub-directory
    split = os.path.split(split_dir)[-1]
    out_dir = os.path.join(out_dir, split)
    create_folders(out_dir)

    # Get list of commands
    commands = parse_script(script, gpus)

    # Move hparams and script files into folder
    if not no_hparams:
        dir_, name = os.path.split(hparams_dir)
        init_project_folder(default_folder=dir_,
                            preset=name,
                            out_folder=out_dir,
                            data_dir=split_dir)

    # Change directory and file permissions
    os.chdir(out_dir)

    # Log
    lock.acquire()
    s = f"[*] Running experiment: {split}"
    delim = '-' * len(s)
    logger.info(f"\n{delim}\n" +
                f"{s}\n" +
                f"Data dir:   {split_dir}\n" +
                f"Out dir:    {out_dir}\n" +
                f"Using GPUs: {gpus}\n" +
                f"Running commands:\n" +
                "\n".join([f" ({i+1}) {' '.join(command)}" for i, command in enumerate(commands)]) + f"\n{delim}")
    lock.release()

    # Run the commands
    run_next_command = True
    for command in commands:
        if not run_next_command:
            break
        str_command = ' '.join(command)
        lock.acquire()
        logger.info(f"[{split} - STARTING] {str_command}")
        lock.release()
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = p.communicate()
        rc = p.returncode
        lock.acquire()
        if rc != 0:
            logger.error(f"[{split} - ERROR - Exit code {rc}] {str_command}\n\n"
                         f"----- START error message -----\n{err.decode('utf-8')}\n"
                         "----- END error message -----\n")
            run_next_command = False
        else:
            logger.info(f"[{split} - FINISHED] {str_command}")
        lock.release()

    # Add the GPUs back into the queue
    gpu_queue.put(gpus)


def start_gpu_monitor_process(args, gpu_queue, gpu_sets):
    procs = []
    if args.monitor_gpus_every is not None and args.monitor_gpus_every:
        logger.info(f"\nOBS: Monitoring GPU pool every {args.monitor_gpus_every} seconds\n")
        # Start a process monitoring new GPU availability over time
        stop_event = Event()
        t = Process(target=monitor_gpus, args=(args.monitor_gpus_every,
                                               gpu_queue,
                                               args.num_gpus,
                                               args.ignore_gpus,
                                               gpu_sets,
                                               stop_event))
        t.start()
        procs.append(t)
    else:
        stop_event = None
    return procs, stop_event


def _assert_run_split(monitor_gpus_every, num_jobs):
    if monitor_gpus_every is not None:
        raise ValueError("--monitor_gpus_every is not a valid argument"
                         " to use with --run_on_split.")
    if num_jobs != 1:
        raise ValueError("--num_jobs is not a valid argument to use with"
                         " --run_on_split.")


def _assert_force_and_ignore_gpus(force_gpus, ignore_gpu):
    force_gpus = gpu_string_to_list(force_gpus)
    ignore_gpu = gpu_string_to_list(ignore_gpu)
    overlap = set(force_gpus) & set(ignore_gpu)
    if overlap:
        raise RuntimeError("Cannot both force and ignore GPU(s) {}. "
                           "Got forced GPUs {} and ignored GPUs {}".format(
            overlap, force_gpus, ignore_gpu
        ))


def prepare_hparams_dir(hparams_dir):
    if not os.path.exists(hparams_dir):
        # Check local hparams.yaml file, move into hparams_dir
        if os.path.exists("hparams.yaml"):
            os.mkdir(hparams_dir)
            hparams = YAMLHParams("hparams.yaml", no_version_control=True)
            for dataset, path in hparams['datasets'].items():
                destination = os.path.join(hparams_dir, path)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.move(path, destination)
            shutil.move("hparams.yaml", hparams_dir)
        else:
            raise RuntimeError("Must specifiy hyperparameters in a folder at path --hparams_prototype_dir <path> OR " + \
                               "have a hparams.yaml file at the current working directory (i.e. project folder)")


def assert_args(args, n_splits):
    # User input assertions
    _assert_force_and_ignore_gpus(args.force_gpus, args.ignore_gpus)
    if args.run_on_split:
        _assert_run_split(args.monitor_gpus_every,
                          args.num_jobs)
        if args.start_from:
            raise RuntimeError("Should not use both --run_on_split and "
                               "--start_from arguments.")
    first_split = args.run_on_split or args.start_from
    if first_split < 0 or first_split > (n_splits-1):
        raise RuntimeError("--run_on_split or --start_from is out of range"
                           " [0-{}] with value {}".format(n_splits-1,
                                                          first_split))


def run(args):
    cv_dir = os.path.abspath(args.cv_dir)
    # Get list of folders of CV data to run on
    cv_folders = get_cv_folders(cv_dir)
    assert_args(args, n_splits=len(cv_folders))
    out_dir = os.path.abspath(args.out_dir)
    hparams_dir = os.path.abspath(args.hparams_prototype_dir)
    prepare_hparams_dir(hparams_dir)
    create_folders(out_dir)

    if args.wait_for:
        # Wait for pid before proceeding
        from utime.utils import await_pids
        await_pids(args.wait_for)
    if args.run_on_split is not None:
        # Run on a single split
        cv_folders = [cv_folders[args.run_on_split]]
    if args.force_gpus:
        # Only these GPUs fill be chosen from
        from utime.utils import set_gpu
        set_gpu(args.force_gpus)
    if args.num_gpus:
        # Get GPU sets (up to the number of splits)
        gpu_sets = get_free_gpu_sets(args.num_gpus,
                                     args.ignore_gpus)[:len(cv_folders)]
    elif not args.num_jobs or args.num_jobs < 0:
        raise ValueError("Should specify a number of jobs to run in parallel "
                         "with the --num_jobs flag when using 0 GPUs pr. "
                         "process (--num_gpus=0 was set).")
    else:
        gpu_sets = ["''"] * args.num_jobs

    # Get process pool, lock and GPU queue objects
    lock = Lock()
    gpu_queue = Queue()
    for gpu in gpu_sets:
        gpu_queue.put(gpu)

    # Get file paths
    script = os.path.abspath(args.script_prototype)

    # Get GPU monitor process
    running_processes, stop_event = start_gpu_monitor_process(args, gpu_queue, gpu_sets)

    try:
        for cv_folder in cv_folders[args.start_from:]:
            gpus = gpu_queue.get()
            t = Process(target=run_sub_experiment,
                        args=(cv_folder, out_dir, script, hparams_dir,
                              args.no_hparams, gpus, gpu_queue, lock))
            t.start()
            running_processes.append(t)
            for t in running_processes:
                if not t.is_alive():
                    t.join()
    except KeyboardInterrupt:
        for t in running_processes:
            t.terminate()
    if stop_event is not None:
        stop_event.set()
    for t in running_processes:
        t.join()


def entry_func(args=None):
    # Get parser
    parser = get_parser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="a")  # Append mode for if --run_on_split support
    run(args)


if __name__ == "__main__":
    entry_func()
