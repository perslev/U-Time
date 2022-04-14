import numpy as np
from sklearn.metrics import cohen_kappa_score


def class_wise_kappa(true, pred, n_classes=None):
    if n_classes is None:
        classes = np.unique(true)
    else:
        classes = np.arange(max(2, n_classes))
    kappa_scores = np.empty(shape=classes.shape, dtype=np.float32)
    kappa_scores.fill(np.nan)
    for idx, _class in enumerate(classes):
        s1 = true == _class
        s2 = pred == _class
        if np.any(s1) or np.any(s2):
            kappa_scores[idx] = cohen_kappa_score(s1, s2)
    return kappa_scores
