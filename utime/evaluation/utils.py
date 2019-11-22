import numpy as np


def _get_tp_rel_sel_from_cm(cm):
    tp = np.diagonal(cm)
    rel = np.sum(cm, axis=0)
    sel = np.sum(cm, axis=1)
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
