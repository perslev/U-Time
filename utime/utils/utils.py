import logging
import os
import subprocess
import time
from collections.abc import Iterable

logger = logging.getLogger(__name__)


def create_folders(folders, create_deep=False):
    def safe_make(path, make_func):
        try:
            make_func(path)
        except FileExistsError:
            # If running many jobs in parallel this may occur
            pass
    make_func = os.mkdir if not create_deep else os.makedirs
    if isinstance(folders, str):
        if not os.path.exists(folders):
            safe_make(folders, make_func)
    else:
        folders = list(folders)
        for f in folders:
            if f is None:
                continue
            if not os.path.exists(f):
                safe_make(f, make_func)


def flatten_lists_recursively(list_of_lists):
    for list_ in list_of_lists:
        if isinstance(list_, Iterable) and not isinstance(list_, (str, bytes)):
            yield from flatten_lists_recursively(list_)
        else:
            yield list_


def highlighted(string):
    length = len(string) if "\n" not in string else max([len(s) for s in string.split("\n")])
    border = "-" * length
    return "%s\n%s\n%s" % (border, string, border)


def await_pids(pids, check_every=120):
    if isinstance(pids, str):
        for pid in pids.split(","):
            wait_for(int(pid), check_every=check_every)
    else:
        wait_for(pids, check_every=check_every)


def wait_for(pid, check_every=120):
    """
    Check for a running process with pid 'pid' and only return when the process
    is no longer running. Checks the process list every 'check_every' seconds.
    """
    if not pid:
        return
    if not isinstance(pid, int):
        try:
            pid = int(pid)
        except ValueError as e:
            raise ValueError(f"Cannot wait for pid '{pid}', must be an integer") from e
    _wait_for(pid, check_every)


def _wait_for(pid, check_every=120):
    still_running = True
    logging.info(f"\n[*] Waiting for process pid={pid} to terminate...")
    while still_running:
        ps = subprocess.Popen(("ps", "-p", f"{pid}"), stdout=subprocess.PIPE)
        try:
            output = subprocess.check_output(("grep", f"{pid}"), stdin=ps.stdout)
        except subprocess.CalledProcessError:
            output = False
        ps.wait()
        still_running = bool(output)
        if still_running:
            logging.info(f"Process {pid} still running... (sleeping {check_every} seconds)")
            time.sleep(check_every)
