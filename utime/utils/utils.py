"""
A set of general utility functions used across the codebase
"""

import numpy as np
from contextlib import contextmanager
import logging


def exactly_one_specified(*inputs):
    """
    Returns True if exactly one of *inputs is None

    Args:
        *inputs: One or more arguments to test against

    Returns:
        bool
    """
    not_none = np.array(list(map(lambda x: x is not None, inputs)))
    return np.sum(not_none) == 1


def b_if_a_is_none(a, b):
    """ Returns 'b' if 'a' is None, otherwise returns 'a' """
    if a is None:
        return b
    else:
        return a


def assert_all_loaded(pairs, raise_=True):
    """
    Returns True if all SleepStudy objects in 'pairs' have the 'loaded'
    property set to True, otherwise returns False.

    If raise_ is True, raises a NotImplementedError if one or more objects are
    not loaded. Otherwise, returns the value of the assessment.

    Temp. until queue functionality implemented
    """
    loaded_pairs = [p for p in pairs if p.loaded]
    if len(loaded_pairs) != len(pairs):
        if raise_:
            raise NotImplementedError("BatchSequence currently requires all"
                                      " samples to be loaded")
        else:
            return False
    return True


def ensure_list_or_tuple(obj):
    """
    Takes some object and wraps it in a list - i.e. [obj] - unless the object
    is already a list or a tuple instance. In that case, simply returns 'obj'

    Args:
        obj: Any object

    Returns:
        [obj] if obj is not a list or tuple, else obj
    """
    return [obj] if not isinstance(obj, (list, tuple)) else obj


@contextmanager
def cd_context(path):
    """
    Context manager for changing directory and back in a context.
    E.g. usage is:

    with cd_context("my_path"):
        ... do something in "my_path"
    ... do something back at original path

    Args:
        path: A string, path

    Returns:
        yields inside the specified directory
    """
    import os
    cur_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cur_dir)


@contextmanager
def mne_no_log_context():
    """ Disables the logger of the mne module inside the context only """
    log = logging.getLogger('mne')
    mem = log.disabled
    log.disabled = True
    try:
        yield
    finally:
        log.disabled = mem
