import logging
import numpy as np
import tensorflow
from typing import List
from tensorflow_addons import optimizers as addon_optimizers
from tensorflow_addons import activations as addon_activations
from tensorflow_addons import losses as addon_losses
from tensorflow_addons import metrics as addon_metrics
from psg_utils.utils import ensure_list_or_tuple
from utime.errors import NotSparseError
from utime.evaluation import loss_functions as custom_loss_functions
from utime.evaluation.utils import ignore_out_of_bounds_classes_wrapper

logger = logging.getLogger(__name__)


def ensure_sparse(loss_and_metric_names: list):
    """
    Checks that 'sparse' is a substring of each string in a list of loss and/or
    metric names. Raises NotSparseError if one or more does not contain the
    substring.
    """
    for i, m in enumerate(loss_and_metric_names):
        if "sparse" not in m.lower():
            # Default error message to raise with non-sparse losses or metrics passed
            raise NotSparseError("This implementation now requires integer targets "
                                 "as opposed to one-hot encoded targets. "
                                 "All metrics and loss functions should be named "
                                 "'sparse_[org_name]' to reflect this in accordance"
                                 " with the naming convention of TensorFlow.keras.")


def _get_classes_or_funcs(string_list: list, func_modules: list) -> List[callable]:
    """
    Helper for 'init_losses' or 'init_metrics'.
    Please refer to their docstrings.

    Args:
        string_list:  (list)   List of strings, each giving a name of a metric
                               or loss to use for training. The name should
                               refer to a function or class in either tf_funcs
                               or custom_funcs modules.
        func_modules: (module or list of modules) A Tensorflow.keras module of losses or metrics,
                                                  or a list of various modules to look through.

    Returns:
        A list of len(string_list) of classes/functions of losses/metrics/optimizers/activation functions etc.
    """
    functions_or_classes = []
    func_modules = ensure_list_or_tuple(func_modules)
    for func_or_class_str in ensure_list_or_tuple(string_list):
        found = False
        for module in func_modules:
            found = getattr(module, func_or_class_str, False)
            if found:
                logger.info(f"Found requested class '{func_or_class_str}' in module '{module}'")
                functions_or_classes.append(found)  # return the first found
                break
        if not found:
            raise AttributeError(f"Did not find loss/metric function {func_or_class_str} "
                                 f"in the module(s) '{func_modules}'")
    return functions_or_classes


def _assert_all_classes(list_of_classes, assert_subclass_of):
    """
    Check that all members of list_of_classes are classes and
    that all members are subclasses of class 'assert_subclass_of'.
    """
    for class_ in ensure_list_or_tuple(list_of_classes):
        if not isinstance(class_, type) or (assert_subclass_of is not None and not issubclass(class_, assert_subclass_of)):
            raise TypeError(
                f"The loss/metric function '{class_}' is not a class or is not a "
                f"subclass of the expected '{assert_subclass_of}' class. All loss & metric functions "
                f"must be classes. For instance, if you specified a keras loss function such as "
                "'sparse_categorical_crossentropy', change this to its corresponding loss class "
                "'SparseCategoricalCrossentropy'. Similarly, for metrics such as "
                "'sparse_categorical_crossentropy' --> 'SparseCategoricalCrossentropy'."
            )


def _init_losses_or_metrics(list_of_losses_or_metrics, ignore_out_of_bounds_classes, wrap_method_name=None, **init_kwargs):
    """
    TODO
    """
    for i, func_or_cls in enumerate(list_of_losses_or_metrics):
        try:
            func_or_cls = func_or_cls(**init_kwargs)
        except TypeError as e:
            if "reduction" in str(e):
                raise TypeError("All loss functions must currently be "
                                "callable and accept the 'reduction' "
                                "parameter specifying a "
                                "tf.keras.losses.Reduction type. If you "
                                "specified a keras loss function such as "
                                "'sparse_categorical_crossentropy', change "
                                "this to its corresponding loss class "
                                "'SparseCategoricalCrossentropy'. If "
                                "you implemented a custom loss function, "
                                "please raise an issue on GitHub.") from e
            else:
                raise e
        if ignore_out_of_bounds_classes:
            if wrap_method_name:
                # Wrap a specific method on the class
                if not hasattr(func_or_cls, wrap_method_name):
                    raise AttributeError(f"Cannot wrap method of name '{wrap_method_name}' on "
                                         f"class '{func_or_cls}'. Class hos n such attribute.")
                wrapped = ignore_out_of_bounds_classes_wrapper(getattr(func_or_cls, wrap_method_name))
                setattr(func_or_cls, wrap_method_name, wrapped)
            else:
                func_or_cls = ignore_out_of_bounds_classes_wrapper(func_or_cls)
        list_of_losses_or_metrics[i] = func_or_cls
    return list_of_losses_or_metrics


def init_losses(loss_string_list, reduction, ignore_out_of_bounds_classes=False, **kwargs):
    """
    Takes a list of strings each naming a loss function to return. The string
    name should correspond to a function or class that is an attribute of
    either the tensorflow.keras.losses or utime.evaluation.losses modules.

    The returned values are either references to the loss functions to use, or
    initialized loss classes for some custom losses (used when the loss
    requires certain parameters to be set).

    Args:
        loss_string_list: (list)   A list of strings each naming a loss to
                                   return
        reduction: (tf.keras.losses.Reduction) TODO
        ignore_out_of_bounds_classes (bool) TODO
        **kwargs:         (dict)   Parameters that will be passed to all class
                                   loss functions (i.e. not to functions)

    Returns:
        A list of length(loss_string_list) of loss functions or initialized
        classes
    """
    losses = _get_classes_or_funcs(loss_string_list,
                                   func_modules=[tensorflow.keras.losses,
                                                 addon_losses,
                                                 custom_loss_functions])
    _assert_all_classes(losses, assert_subclass_of=tensorflow.keras.losses.Loss)
    return _init_losses_or_metrics(losses,
                                   reduction=reduction,
                                   ignore_out_of_bounds_classes=ignore_out_of_bounds_classes,
                                   wrap_method_name='call',
                                   **kwargs)


def init_metrics(metric_string_list, ignore_out_of_bounds_classes=False, **kwargs):
    """
    Same as 'init_losses', but for metrics.
    Please refer to the 'init_losses' docstring.
    """
    metrics = _get_classes_or_funcs(metric_string_list,
                                    func_modules=[tensorflow.keras.metrics,
                                                  addon_metrics])
    _assert_all_classes(metrics, assert_subclass_of=tensorflow.keras.metrics.Metric)
    return _init_losses_or_metrics(metrics,
                                   ignore_out_of_bounds_classes=ignore_out_of_bounds_classes,
                                   wrap_method_name='update_state',
                                   **kwargs)


def init_optimizer(optimizer_string, **kwargs):
    """
    Same as 'init_losses', but for optimizers.
    Please refer to the 'init_losses' docstring.
    """
    optimizer = _get_classes_or_funcs(
        optimizer_string,
        func_modules=[tensorflow.keras.optimizers, addon_optimizers]
    )
    assert len(optimizer) == 1, f'Received unexpected number of optimizers ({len(optimizer)}, expected 1)'
    return optimizer[0](**kwargs)


def get_activation_function(activation_string):
    """
    Same as 'init_losses', but for optimizers.
    Please refer to the 'init_losses' docstring.
    """
    activation = _get_classes_or_funcs(
        activation_string,
        func_modules=[tensorflow.keras.activations, addon_activations]
    )
    assert len(activation) == 1, f'Received unexpected number of activation functions ({len(activation)}, expected 1)'
    return activation[0]


def get_steps(samples_per_epoch, sequence):
    """
    Computes the number of gradient update steps to use for training or
    validation.

    Takes an integer 'samples_per_epoch' specifying how many samples should be
    used in 1 epoch. Returns the (ceiled) number of batches of size
    'batch_size' needed for such epoch.

    If 'samples_per_epoch' is None, returns the length of the
    Sequence object.

    Args:
        samples_per_epoch: (int)      Number of samples to use in an epoch
        sequence:          (Sequence) The Sequence object from which samples
                                      will be generated

    Returns:
        (int) Number of steps to take in the epoch
    """
    if samples_per_epoch:
        return int(np.ceil(samples_per_epoch / sequence.batch_size))
    else:
        return len(sequence)
