import logging
import numpy as np
import tensorflow as tf
from functools import wraps

logger = logging.getLogger(__name__)


def ignore_out_of_bounds_classes_wrapper(func):
    """
    For a model that outputs 'n_pred_classes' classes, this wrapper removes entries in the
    true/pred pairs for which the true label is not in the range [0, 1, ..., n_pred_classes - 1].
    'n_pred_classes' is determined as the length of the prediction tensor on the last dimension.
    """
    @wraps(func)
    @tf.function
    def wrapper(true, pred, **kwargs):
        true.set_shape(pred.get_shape()[:-1] + [1])
        n_pred_classes = pred.get_shape()[-1]
        true = tf.reshape(true, [-1])
        pred = tf.reshape(pred, [-1, n_pred_classes])
        mask = tf.where(tf.logical_and(
                            tf.greater_equal(true, 0),
                            tf.less(true, n_pred_classes)
                        ),
                        tf.ones_like(true),
                        tf.zeros_like(true))
        mask = tf.cast(mask, tf.bool)
        true = tf.boolean_mask(true, mask, axis=0)
        pred = tf.boolean_mask(pred, mask, axis=0)
        return func(true, pred, **kwargs)
    logger.info(f"Wrapping loss/metric function '{func}' to ignore 'true' "
                f"classes with integer values outside of the model prediction integer range "
                f"(e.g., ignoring true class labels of value 5 or -1 if n_classes=5, "
                f"i.e. when model outputs values in [0, 1, 2, 3, 4]).")
    return wrapper


def _get_tp_rel_sel_from_cm(cm):
    tp = np.diagonal(cm)
    sel = np.sum(cm, axis=0)
    rel = np.sum(cm, axis=1)
    return tp, rel, sel


def f1_scores_from_cm(cm):
    precisions = precision_scores_from_cm(cm)
    recalls = recall_scores_from_cm(cm)

    # prepare arrays
    dices = np.zeros_like(precisions)

    # Compute dice
    intrs = (2 * precisions * recalls)
    union = (precisions + recalls)
    dice_mask = union > 0
    dices[dice_mask] = intrs[dice_mask] / union[dice_mask]
    return dices


def precision_scores_from_cm(cm):
    tp, rel, sel = _get_tp_rel_sel_from_cm(cm)
    sel_mask = sel > 0
    precisions = np.zeros(shape=tp.shape, dtype=np.float32)
    precisions[sel_mask] = tp[sel_mask] / sel[sel_mask]
    return precisions


def recall_scores_from_cm(cm):
    tp, rel, sel = _get_tp_rel_sel_from_cm(cm)
    rel_mask = rel > 0
    recalls = np.zeros(shape=tp.shape, dtype=np.float32)
    recalls[rel_mask] = tp[rel_mask] / rel[rel_mask]
    return recalls


def concatenate_true_pred_pairs(pairs=None, trues=None, pred=None):
    if pairs is None and (trues is None or pred is None):
        raise ValueError("Must specify either 'pairs' argument "
                         "or both 'trues' and 'pred' arguments")
    if pairs is not None:
        trues, pred = zip(*pairs)
    return np.concatenate(list(trues)), np.concatenate(list(pred))
