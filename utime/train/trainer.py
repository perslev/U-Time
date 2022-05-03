"""
The Trainer class prepares and launches training of a model.
Most importantly, it compiles the tf.keras Model object with according to the
specified optimizer, loss and metrics and implements the .fit method for
training the model given a set of parameters and (non-initialized) callbacks.
"""

import logging
import tensorflow as tf
from tensorflow.python.framework.errors_impl import ResourceExhaustedError, InternalError
from utime.callbacks import init_callback_objects, remove_validation_callbacks
from utime.callbacks import Validation, LearningCurve, MeanReduceLogArrays, PrintDividerLine, MemoryConsumption
from psg_utils.utils import ensure_list_or_tuple
from utime.train.utils import ensure_sparse, init_losses, init_metrics, init_optimizer, get_steps

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    Handles initialization and logging of model fitting sessions.
    """
    def __init__(self, model):
        """
        Args:
            model:      (tf.keras Model) Initialized model to train
        """
        self.model = model

    def compile_model(self, optimizer, loss, metrics, reduction,
                      ignore_out_of_bounds_classes=False, check_sparse=False,
                      optimizer_kwargs={}, loss_kwargs={}, metric_kwargs={}, **kwargs):
        """
        Compile the stored tf.keras Model instance stored in self.model
        Sets the loss function, optimizer and metrics

        Args:
            optimizer:        (string) The name of a tf.keras.optimizers Optimizer
            loss:             (string) The name of a tf.keras.losses or
            metrics:          (list)   List of tf.keras.metrics or
                                       MultiPlanarUNet metrics.
            reduction:                          TODO
            check_sparse:                       TODO
            ignore_out_of_bounds_classes (bool) TODO
            optimizer_kwargs: (dict)   Key-word arguments passed to the Optimizer
            loss_kwargs:      (dict)   Key-word arguments passed to all Loss functions
            metric_kwargs:    (dict)   Key-word arguments passed to all Metrics functions
                                       MultiPlanarUnet loss function
            **kwargs:         (dict)   Key-word arguments passed to losses
                                       and/or metrics that accept such.
        """
        # Make sure sparse metrics and loss are specified as sparse
        metrics = ensure_list_or_tuple(metrics)
        losses = ensure_list_or_tuple(loss)
        if check_sparse:
            ensure_sparse(metrics+losses)

        # Initialize optimizer, loss(es) and metric(s) from tf.keras, tf addons or utime.evaluation
        optimizer = init_optimizer(optimizer, **optimizer_kwargs)
        losses = init_losses(losses, reduction, ignore_out_of_bounds_classes, **loss_kwargs)
        metrics = init_metrics(metrics, ignore_out_of_bounds_classes, **metric_kwargs)

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        logger.info(f"\nOptimizer:   {optimizer}\n"
                    f"Loss funcs:  {losses}\n"
                    f"Metrics:     {metrics}")
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
                logger.error(f"[MEMORY ERROR] Reducing batch size by 2 (now {batch_size})")
                if batch_size < 1:
                    logger.error("[ERROR] Batch size negative or zero! Stopping training.")
                    fitting = False
            except KeyboardInterrupt:
                fitting = False
            except Exception as e:
                logger.exception(str(e), exc_info=e)
                raise e
        logger.info("Training stopped.")
        return self.model

    def _fit(self,
             train,
             val,
             batch_size,
             n_epochs,
             callbacks,
             train_samples_per_epoch,
             max_val_studies_per_dataset,
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
        logger.info(f"Using {train_steps} steps per train epoch")

        if val is None:
            # No validation to be performed, remove callbacks that might need
            # validation data to function properly
            remove_validation_callbacks(callbacks)
        else:
            val.batch_size = batch_size
            # Add validation callback
            # Important: Should be first in callbacks list as other CBs may
            # depend on the validation metrics/loss
            callbacks = [Validation(val, max_val_studies_per_dataset), MeanReduceLogArrays()] + callbacks

        # Add various callbacks for plotting learning curves etc.
        callbacks.append(LearningCurve())
        # callbacks.append(MemoryConsumption(max_gib=45))
        # callbacks.append(CarbonUsageTracking(epochs=n_epochs, add_to_logs=False))

        # Get initialized callback objects
        callbacks = [PrintDividerLine()] + callbacks + [PrintDividerLine()]
        callbacks, cb_dict = init_callback_objects(callbacks)

        # Wrap generator in TF Dataset and disable auto shard
        dtypes, shapes = list(zip(*map(lambda x: (x.dtype, x.shape), train[0])))
        train = tf.data.Dataset.from_generator(train, dtypes, shapes)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train = train.with_options(options)

        # Fit the model
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
