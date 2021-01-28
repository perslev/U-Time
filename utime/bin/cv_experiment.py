import os
from multiprocessing import Process, Lock, Queue, Event
from mpunet.utils import create_folders
from mpunet.logging import Logger
from utime.bin.init import init_project_folder
import argparse
import subprocess


def get_parser():
    parser = argparse.ArgumentParser(description="Prepare a data folder for a"
                                                 "CV experiment setup.")
    parser.add_argument("--CV_dir", type=str, required=True,
                        help="Directory storing split subfolders as output by"
                             " cv_split.py")
    parser.add_argument("--out_dir", type=str, default="./splits",
                        help="Folder in which experiments will be run and "
                             "results stored.")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use per process. This also "
                             "defines the number of parallel jobs to run.")
    parser.add_argument("--force_GPU", type=str, default="",
                        help="A list of one or more GPU IDs "
                             "(comma separated) from which GPU resources "
                             "will supplied to each split, independent of"
                             " the current memory usage of the GPUs.")
    parser.add_argument("--ignore_GPU", type=str, default="",
                        help="A list of one or more GPU IDs "
                             "(comma separated) that will not be considered.")
    parser.add_argument("--num_jobs", type=int, default=1,
                        help="OBS: Only in effect when --num_GPUs=0. Sets"
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
                        help="Waiting for PID to terminate before starting "
                             "training process.")
    parser.add_argument("--monitor_GPUs_every", type=int, default=None,
                        help="If specified, start a background process which"
                             " monitors every 'monitor_GPUs_every' seconds "
                             "whether new GPUs have become available than may"
                             " be included in the CV experiment GPU resource "
                             "pool.")
    return parser


def get_CV_folders(dir):
    key = lambda x: int(x.split("_")[-1])
    return [os.path.join(dir, p) for p in sorted(os.listdir(dir), key=key)]


def _get_GPU_sets(free_gpus, num_GPUs):
    free_gpus = list(map(str, free_gpus))
    return [",".join(free_gpus[x:x + num_GPUs]) for x in range(0, len(free_gpus),
                                                               num_GPUs)]


def get_free_GPU_sets(num_GPUs, ignore_gpus=None):
    from mpunet.utils.system import GPUMonitor
    mon = GPUMonitor()
    ignore_gpus = _gpu_string_to_list(ignore_gpus or "", as_int=True)
    free_gpus = sorted(mon.free_GPUs, key=lambda x: int(x))
    mon.stop()
    free_gpus = list(filter(lambda gpu: gpu not in ignore_gpus, free_gpus))
    total_GPUs = len(free_gpus)

    if total_GPUs % num_GPUs or not free_gpus:
        if total_GPUs < num_GPUs:
            raise ValueError("Invalid number of GPUs per process '%i' for total "
                             "GPU count of '%i' - must be evenly divisible." %
                             (num_GPUs, total_GPUs))
        else:
            raise NotImplementedError
    else:
        return _get_GPU_sets(free_gpus, num_GPUs)


def monitor_GPUs(every, gpu_queue, num_GPUs, ignore_GPU, current_pool, stop_event):
    import time
    # Make flat version of the list of gpu sets
    current_pool = [gpu for sublist in current_pool for gpu in sublist.split(",")]
    while not stop_event.is_set():
        # Get available GPU sets. Will raise ValueError if no full set is
        # available
        try:
            gpu_sets = get_free_GPU_sets(num_GPUs, ignore_GPU)
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


def parse_script(script, GPUs):
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
                cmd.append("--force_GPU=%s" % GPUs)
            commands.append(cmd)
    return commands


def run_sub_experiment(split_dir, out_dir, script, hparams_dir, no_hparams,
                       GPUs, GPU_queue, lock, logger):

    # Create sub-directory
    split = os.path.split(split_dir)[-1]
    out_dir = os.path.join(out_dir, split)
    create_folders(out_dir)

    # Get list of commands
    commands = parse_script(script, GPUs)

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
    s = "[*] Running experiment: %s" % split
    logger("\n%s\n%s" % ("-" * len(s), s))
    logger("Data dir:", split_dir)
    logger("Out dir:", out_dir)
    logger("Using GPUs:", GPUs)
    logger("\nRunning commands:")
    for i, command in enumerate(commands):
        logger(" %i) %s" % (i+1, " ".join(command)))
    logger("-"*len(s))
    lock.release()

    # Run the commands
    run_next_command = True
    for command in commands:
        if not run_next_command:
            break
        lock.acquire()
        logger("[%s - STARTING] %s" % (split, " ".join(command)))
        lock.release()
        p = subprocess.Popen(command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        _, err = p.communicate()
        rc = p.returncode
        lock.acquire()
        if rc != 0:
            logger("[%s - ERROR - Exit code %i] %s" % (split, rc, " ".join(command)))
            logger("\n----- START error message -----\n%s\n"
                   "----- END error message -----\n" % err.decode("utf-8"))
            run_next_command = False
        else:
            logger("[%s - FINISHED] %s" % (split, " ".join(command)))
        lock.release()

    # Add the GPUs back into the queue
    GPU_queue.put(GPUs)


def _gpu_string_to_list(gpu_list, as_int=False):
    str_gpus = list(filter(None, gpu_list.replace(" ", "").split(",")))
    if as_int:
        return list(map(int, str_gpus))
    return str_gpus


def start_gpu_monitor_process(args, gpu_queue, gpu_sets, logger):
    procs = []
    if args.monitor_GPUs_every is not None and args.monitor_GPUs_every:
        logger(
            "\nOBS: Monitoring GPU pool every %i seconds\n" % args.monitor_GPUs_every)
        # Start a process monitoring new GPU availability over time
        stop_event = Event()
        t = Process(target=monitor_GPUs, args=(args.monitor_GPUs_every,
                                               gpu_queue,
                                               args.num_GPUs, args.ignore_GPU,
                                               gpu_sets, stop_event))
        t.start()
        procs.append(t)
    else:
        stop_event = None
    return procs, stop_event


def _assert_run_split(monitor_GPUs_every, num_jobs):
    if monitor_GPUs_every is not None:
        raise ValueError("--monitor_GPUs_every is not a valid argument"
                         " to use with --run_on_split.")
    if num_jobs != 1:
        raise ValueError("--num_jobs is not a valid argument to use with"
                         " --run_on_split.")


def _assert_force_and_ignore_gpus(force_gpu, ignore_gpu):
    force_gpu = _gpu_string_to_list(force_gpu)
    ignore_gpu = _gpu_string_to_list(ignore_gpu)
    overlap = set(force_gpu) & set(ignore_gpu)
    if overlap:
        raise RuntimeError("Cannot both force and ignore GPU(s) {}. "
                           "Got forced GPUs {} and ignored GPUs {}".format(
            overlap, force_gpu, ignore_gpu
        ))


def assert_args(args, n_splits):
    # User input assertions
    _assert_force_and_ignore_gpus(args.force_GPU, args.ignore_GPU)
    if args.run_on_split:
        _assert_run_split(args.monitor_GPUs_every,
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
    cv_dir = os.path.abspath(args.CV_dir)
    # Get list of folders of CV data to run on
    cv_folders = get_CV_folders(cv_dir)
    assert_args(args, n_splits=len(cv_folders))
    out_dir = os.path.abspath(args.out_dir)
    hparams_dir = os.path.abspath(args.hparams_prototype_dir)
    create_folders(out_dir)

    # Wait for PID?
    if args.wait_for:
        from mpunet.utils import await_PIDs
        await_PIDs(args.wait_for)

    if args.run_on_split is not None:
        cv_folders = [cv_folders[args.run_on_split]]
        log_appendix = "_split{}".format(args.run_on_split)
    else:
        log_appendix = ""

    # Get a logger object
    logger = Logger(base_path="./", active_file="output" + log_appendix,
                    print_calling_method=False, overwrite_existing=True)

    if args.force_GPU:
        # Only these GPUs fill be chosen from
        from mpunet.utils import set_gpu
        set_gpu(args.force_GPU)
    if args.num_GPUs:
        # Get GPU sets (up to the number of splits)
        gpu_sets = get_free_GPU_sets(args.num_GPUs,
                                     args.ignore_GPU)[:len(cv_folders)]
    elif not args.num_jobs or args.num_jobs < 0:
        raise ValueError("Should specify a number of jobs to run in parallel "
                         "with the --num_jobs flag when using 0 GPUs pr. "
                         "process (--num_GPUs=0 was set).")
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
    running_processes, stop_event = start_gpu_monitor_process(args, gpu_queue,
                                                              gpu_sets, logger)

    try:
        for cv_folder in cv_folders[args.start_from:]:
            gpus = gpu_queue.get()
            t = Process(target=run_sub_experiment,
                        args=(cv_folder, out_dir, script, hparams_dir,
                              args.no_hparams, gpus, gpu_queue,
                              lock, logger))
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
    run(parser.parse_args(args))


if __name__ == "__main__":
    entry_func()
