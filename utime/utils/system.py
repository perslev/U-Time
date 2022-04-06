import logging
import re
import os
import numpy as np
from time import sleep
from subprocess import check_output

logger = logging.getLogger(__name__)


def _get_system_wide_set_gpus():
    allowed_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allowed_gpus:
        allowed_gpus = allowed_gpus.replace(" ", "").split(",")
    return allowed_gpus


def gpu_string_to_list(gpu_visibility_string, as_int=False):
    gpus = re.findall(r"(\d+)", str(gpu_visibility_string))
    if as_int:
        gpus = list(map(int, gpus))
    return gpus


def get_free_gpus(max_allowed_mem_usage=400):
    # Check if allowed GPUs are set in CUDA_VIS_DEV.
    allowed_gpus = _get_system_wide_set_gpus()
    if allowed_gpus:
        logger.info(f"[OBS] Considering only system-wise allowed GPUs: {allowed_gpus} (set in"
                    f" CUDA_VISIBLE_DEVICES env variable).")
        return allowed_gpus
    # Else, check GPUs on the system and assume all non-used (mem. use less
    # than max_allowed_mem_usage) is fair game.
    try:
        # Get list of GPUs
        gpu_list = check_output(["nvidia-smi", "-L"], universal_newlines=True)
        gpu_ids = np.array(re.findall(r"GPU[ ]+(\d+)", gpu_list), dtype=np.int)

        # Query memory usage stats from nvidia-smi
        output = check_output(["nvidia-smi", "-q", "-d", "MEMORY"], universal_newlines=True)

        # Fetch the memory usage of each GPU
        mem_usage = re.findall(r"FB Memory Usage.*?Used[ ]+:[ ]+(\d+)",
                               output, flags=re.DOTALL)
        assert len(gpu_ids) == len(mem_usage)

        # Return all GPU ids for which the memory usage is below or eq. to max allowed
        free = list(map(lambda x: int(x) <= max_allowed_mem_usage or 0, mem_usage))
        return list(gpu_ids[free])
    except FileNotFoundError as e:
        raise FileNotFoundError("[ERROR] nvidia-smi is not installed. "
                                "Consider setting the --num_gpus=0 flag.") from e


def _get_gpu_visibility_string(free_gpus: list, num_gpus=1):
    visibility_string = ",".join(map(str, free_gpus[:num_gpus]))
    return visibility_string


def _get_free_gpus_visibility_string(num_gpus=1, max_allowed_mem_usage=400):
    free = get_free_gpus(max_allowed_mem_usage)
    if not free or num_gpus > len(free):
        raise ResourceWarning(f"Requested N={num_gpus} GPUs, but only found {len(free)} GPUs available with memory "
                              f"loads less than or equal to {max_allowed_mem_usage} MiB "
                              f"('None' signals no memory requirement)")
    return _get_gpu_visibility_string(free, num_gpus=num_gpus)


def get_visible_gpus(as_list=True):
    gpu_vis_string = os.environ["CUDA_VISIBLE_DEVICES"].strip(", ")
    if as_list:
        return list(filter(None, map(lambda s: s.strip(), gpu_vis_string.split(","))))
    else:
        return gpu_vis_string


def set_gpu(gpu_visibility_string: str):
    gpu_visibility_string = ",".join(gpu_string_to_list(gpu_visibility_string))
    logger.info(f"Setting CUDA_VISIBLE_DEVICES = '{gpu_visibility_string}'")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_visibility_string


def await_and_set_free_gpu(num_gpus=1, sleep_seconds=60, max_allowed_mem_usage=400, timeout_seconds=3600):
    gpu_vis_string = ""
    wait_seconds = 0
    if num_gpus > 0:
        logger.info(f"Waiting for free N={num_gpus} GPU(s)...")
        while not gpu_vis_string:
            if wait_seconds >= timeout_seconds:
                raise OSError(f"Could not find {num_gpus} with max allowed memory usage "
                              f"of {max_allowed_mem_usage} MiB within timeout of {timeout_seconds} seconds.")
            try:
                gpu_vis_string = _get_free_gpus_visibility_string(num_gpus, max_allowed_mem_usage)
            except ResourceWarning as e:
                logger.warning(f"Not enough available GPUs. Original warning: {str(e)}")
                logger.warning(f"No available GPUs... Sleeping {sleep_seconds}s "
                               f"(current wait time is {wait_seconds}s. Timeout is {timeout_seconds}s).")
                sleep(sleep_seconds)
                wait_seconds += sleep_seconds
    set_gpu(gpu_vis_string)


def find_and_set_gpus(num_gpus=None, force_gpus=None):
    """
    Utility function to either look for free GPUs and set them visible,
    or set a forced set of GPUs visible.

    Specifically, if force_gpus is specified, set the visibility accordingly,
    count the number of GPUs set and return this number.
    If not, use num_gpus currently available GPUs

    Args:
        num_gpus:  (int)        Number of free/available GPUs to automatically
                                select using 'gpu_mon' when 'force_GPU' is not set.
        force_gpus: (string)    A CUDA_VISIBLE_DEVICES type string to be set

    Returns:
        (int) The number of GPUs now visible
    """
    if num_gpus is None and force_gpus is None:
        raise ValueError("Must specify at least one of 'num_gpus' and 'force_gpus'")
    if not force_gpus:
        await_and_set_free_gpu(num_gpus)
    else:
        set_gpu(force_gpus)
    return len(get_visible_gpus(as_list=True))
