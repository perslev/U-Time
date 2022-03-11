import logging
from tensorflow.keras import callbacks as tfcb
from utime import callbacks as tcb
from utime.callbacks import DelayedCallback

logger = logging.getLogger(__name__)


def init_callback_objects(callbacks):
    """
    Initialize a list of tf.keras/custom callback descriptors.

    Args:
        callbacks: List of callback descriptions. Each list entry should be
                   either a dictionary of the format
                   {"class_name": <name_of_CB_class>,
                   "kwargs": {<dict_with_kwargs>}}
                   ... or an already initialized callback object
                   (which will be skipped).

    Returns:
        A list of initialized callbacks
    """
    cb_objs = []
    cb_dict = {}
    for i, callback in enumerate(callbacks):
        if not isinstance(callback, dict):
            # CB already initialized
            cb = callback
            kwargs = {"params": "?"}
            cls_name = callback.__class__.__name__
            start_from = 0
        else:
            kwargs = callback["kwargs"]
            cls_name = callback["class_name"]
            start_from = callback.get("start_from")
            try:
                cb = getattr(tfcb, cls_name, None) or getattr(tcb, cls_name)
            except AttributeError as e:
                raise ValueError("No callback named %s" % cls_name) from e
            cb = cb(**kwargs)
        if start_from:
            logger.info(f"OBS: '{cls_name}' activates at epoch {start_from}")
            cb = DelayedCallback(callback=cb, start_from=start_from)
        cb_objs.append(cb)
        cb_dict[cls_name] = cb
        logger.info(f"[{i+1}] Using callback: "
                    f"{cb.__class__.__name__}({', '.join([f'{a}={kwargs[a]}' for a in kwargs])})")
    return cb_objs, cb_dict


def remove_validation_callbacks(callbacks):
    """
    Removes all callbacks that rely on validation data

    Takes a list of uninitialized callbacks data, enumerates them and removes
    each entry if one or more of its parameters in 'kwargs' mentions 'val'.

    Args:
        callbacks: A list of dictionaries, each representing a callback

    Returns:
        None, operates in-place
    """
    for i, callback in enumerate(callbacks):
        val_dependent_params = []
        for param in callback["kwargs"].values():
            val_dependent_params.append("val" in str(param).lower())
        if any(val_dependent_params):
            logger.info(f"Removing callback with parameters: {callback} (needs validation data)")
            callbacks.pop(i)
