"""
The Trainer class prepares and launches training of a model.
Most importantly, it compiles the tf.keras Model object with according to the
specified optimizer, loss and metrics and implements the .fit method for
training the model given a set of parameters and (non-initialized) callbacks.
"""

import tensorflow as tf
from tensorflow.python.framework.errors_impl import (ResourceExhaustedError,
                                                     InternalError)
from mpunet.callbacks import (init_callback_objects,
                              remove_validation_callbacks)
from mpunet.logging import ScreenLogger
from mpunet.callbacks import (DividerLine, LearningCurve, MeanReduceLogArrays)
from mpunet.utils import ensure_list_or_tuple
from mpunet.train.utils import (ensure_sparse,
                                init_losses,
                                init_metrics,
                                init_optimizer)
from utime.callbacks import Validation, MemoryConsumption
from utime.train.utils import get_steps


def ignore_class_wrapper(loss_func, n_pred_classes, logger):
    """
    For a model that outputs K classes, this wrapper removes entries in the
    true/pred pairs for which the true label is of integer value K.

    TODO
    """
    @tf.function
    def wrapper(true, pred):
        true.set_shape(pred.get_shape()[:-1] + [1])
        true = tf.reshape(true, [-1])
        pred = tf.reshape(pred, [-1, n_pred_classes])
        mask = tf.where(tf.not_equal(true, n_pred_classes), tf.ones_like(true), tf.zeros_like(true))
        mask = tf.cast(mask, tf.bool)
        true = tf.boolean_mask(true, mask, axis=0)
        pred = tf.boolean_mask(pred, mask, axis=0)
        return loss_func(true, pred)
    logger("Regarding loss func: {}. "
           "Model outputs {} classes; Ignoring class with "
           "integer values {}".format(loss_func, n_pred_classes,
                                      n_pred_classes))
    return wrapper


class Trainer(object):
    """
    Handles initialization and logging of model fitting sessions.
    """
    def __init__(self, model, org_model=None, logger=None):
        """
        Init. simply accepts a model and stores it.
        Optionally, an 'org_model' (original model) may be passed and stored
        as well. This is for training multi-GPU models prepared by the
        tf.keras.utils.multi_gpu_model utility, which returns a new, split
        model for training (passed as 'model' parameter here). For properly
        saving the model parameter, however, it is recommended to use the
        original, non-split model (here passed as 'org_model').

        Args:
            model:      (tf.keras Model) Initialized model to train
            org_model:  (tf.keras Model) Optional single-GPU version for the
                                         passed 'model' parameter.
            logger:     (Logger)         Optional Logger instance
        """
        self.model = model
        self.logger = logger if logger is not None else ScreenLogger()

        # Extra reference to original (non multiple-GPU) model
        # May also be set from a script at a later time (before self.fit call)
        self.org_model = org_model

    def compile_model(self, optimizer, loss, metrics, reduction,
                      ignore_class_int=None, check_sparse=False,
                      optimizer_kwargs={}, loss_kwargs={}, **kwargs):
        """
        Compile the stored tf.keras Model instance stored in self.model
        Sets the loss function, optimizer and metrics

        Args:
            optimizer:        (string) The name of a tf.keras.optimizers Optimizer
            optimizer_kwargs: (dict)   Key-word arguments passed to the Optimizer
            loss:             (string) The name of a tf.keras.losses or
                                       MultiPlanarUnet loss function
            metrics:          (list)   List of tf.keras.metrics or
                                       MultiPlanarUNet metrics.
            reduction         TODO
            check_sparse:     TODO
            **kwargs:         (dict)   Key-word arguments passed to losses
                                       and/or metrics that accept such.
        """
        # Make sure sparse metrics and loss are specified as sparse
        metrics = ensure_list_or_tuple(metrics)
        losses = ensure_list_or_tuple(loss)
        if check_sparse:
            ensure_sparse(metrics+losses)

        # Initialize optimizer, loss(es) and metric(s) from tf.keras or MultiPlanarUNet
        optimizer = init_optimizer(optimizer, self.logger, **optimizer_kwargs)
        losses = init_losses(losses, self.logger, **kwargs)
        for i, loss in enumerate(losses):
            try:
                losses[i] = loss(reduction=reduction, **loss_kwargs)
            except (ValueError, TypeError):
                raise TypeError("All loss functions must currently be "
                                "callable and accept the 'reduction' "
                                "parameter specifying a "
                                "tf.keras.losses.Reduction type. If you "
                                "specified a keras loss function such as "
                                "'sparse_categorical_crossentropy', change "
                                "this to its corresponding loss class "
                                "'SparseCategoricalCrossentropy'. If "
                                "you implemented a custom loss function, "
                                "please raise an issue on GitHub.")
            if ignore_class_int is not None:
                # Mask out class
                losses[i] = ignore_class_wrapper(losses[i], ignore_class_int, self.logger)
        metrics = init_metrics(metrics, self.logger, **kwargs)

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        self.logger("Optimizer:   %s" % optimizer)
        self.logger("Loss funcs:  %s" % losses)
        self.logger("Metrics:     %s" % init_metrics)
        return self

    def fit(self, batch_size, **fit_kwargs):
        """
        Fit the stored tf.keras Model (self.model) on a set of data.

        The 'fit' method is a wrapper around the hidden '_fit' method. It
        handles KeyboardInterrupts (--> stopping training prematurely), TF
        GPU memory errors (--> batch_size is reduced by 2 and training
        restarted), and other exceptions (--> error logged and training
        terminated).

        Please refer to the self._fit method for 'fit_kwargs' argument details.

        Args:
            batch_size: (int)  The initial batch size to run training with
            fit_kwargs: (dict) Keyword arguments passed to self._fit
        """
        fitting = True
        while fitting:
            try:
                self._fit(batch_size=batch_size, **fit_kwargs)
                fitting = False
            except (ResourceExhaustedError, InternalError):
                # Reduce batch size
                batch_size -= 2
                self.logger("\n\n[MEMORY ERROR] Reducing batch size "
                            "by 2 (now %i)" % batch_size)
                if batch_size < 1:
                    self.logger("[ERROR] Batch size negative or zero!")
                    fitting = False
            except KeyboardInterrupt:
                fitting = False
            except Exception as e:
                self.logger(e)
                raise e
        self.logger.print_calling_method = True
        self.logger("Training stopped.")
        return self.model

    def _fit(self,
             train,
             val,
             batch_size,
             n_epochs,
             callbacks,
             train_samples_per_epoch,
             verbose=1,
             init_epoch=0,
             **unused):
        """
        Args:
            train: (Sequence)       The training Sequence object
            val    (Sequence, None) The validation Sequence object or None if no
                                    validation is to be performed
            batch_size: (int)       The batch size to use for training
            n_epochs: (int)         Number of epochs to train for
            callbacks: (list)       List of uninitialized callback kwargs.
            train_samples_per_epoch: (int) Number of training samples to sample
                                           before an epoch is determined over.
            verbose: (int/bool)     Verbosity level passed to keras.fit_generator
            init_epoch: (int)       The initial epoch
            use_multiprocessing: (bool) Whether to use multiprocessing instead
                                        of multithreading.
        """
        train.batch_size = batch_size
        train_steps = get_steps(train_samples_per_epoch, train)
        self.logger("Using {} steps per train epoch".format(train_steps))

        if val is None:
            # No validation to be performed, remove callbacks that might need
            # validation data to function properly
            remove_validation_callbacks(callbacks, self.logger)
        else:
            val.batch_size = batch_size
            # Add validation callback
            # Important: Should be first in callbacks list as other CBs may
            # depend on the validation metrics/loss
            validation = Validation(val, logger=self.logger, verbose=verbose)
            callbacks = [validation, MeanReduceLogArrays()] + callbacks

        # Add various callbacks for plotting learning curves etc.
        callbacks.append(MemoryConsumption(max_gib=45, logger=self.logger))
        callbacks.append(LearningCurve(logger=self.logger))
        # callbacks.append(CarbonUsageTracking(epochs=n_epochs, add_to_logs=False))
        callbacks.append(DividerLine(self.logger))

        # Get initialized callback objects
        callbacks, cb_dict = init_callback_objects(callbacks, self.logger)

        # If ModelCheckPointClean is used, set the original model to store
        # the correct weights when using multi-GPU models
        cb = cb_dict.get("ModelCheckPointClean")
        if cb:
            cb.org_model = self.org_model

        # Temporary memory leak fix
        import tensorflow as tf
        dtypes, shapes = list(zip(*map(lambda x: (x.dtype, x.shape), train[0])))
        train = tf.data.Dataset.from_generator(train, dtypes, shapes)

        # Fit the model
        self.logger.active_log_file = "training"
        self.logger.print_calling_method = False
        self.model.fit(
            train,
            steps_per_epoch=train_steps,
            epochs=n_epochs,
            callbacks=callbacks,
            initial_epoch=init_epoch,
            use_multiprocessing=False,
            workers=3,
            max_queue_size=10,
            shuffle=False,  # Determined by the chosen Sequence class
            verbose=verbose
        )
