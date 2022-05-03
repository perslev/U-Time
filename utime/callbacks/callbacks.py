import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from carbontracker.tracker import CarbonTracker
from tensorflow.keras.callbacks import Callback
from psg_utils.utils import get_memory_usage
from utime.utils import highlighted
from collections import defaultdict
from datetime import timedelta, datetime
from utime.utils.plotting import plot_all_training_curves

logger = logging.getLogger(__name__)


class Validation(Callback):
    """
    Validation computation callback.

    Samples a number of validation batches from a deepsleep
    ValidationMultiSequence object
    and computes for all tasks:
        - Batch-wise validation loss
        - Batch-wise metrics as specified in model.metrics_tensors
        - Epoch-wise pr-class and average precision
        - Epoch-wise pr-class and average recall
        - Epoch-wise pr-class and average dice coefficients
    ... and adds all results to the log dict

    Note: The purpose of this callback over the default tf.keras evaluation
    mechanism is to calculate certain metrics over the entire epoch of data as
    opposed to averaged batch-wise computations.
    """
    def __init__(self, val_sequence, max_val_studies_per_dataset=20):
        """
        Args:
            val_sequence: A deepsleep ValidationMultiSequence object
        """
        super().__init__()
        self.sequences = val_sequence.sequences
        self.max_studies = max_val_studies_per_dataset
        self.n_classes = val_sequence.n_classes
        self.IDs = val_sequence.IDs
        self.print_round = 3
        self.log_round = 4
        self._supports_tf_logs = True

    def _compute_counts(self, pred, true):
        # Argmax and CM elements
        pred = pred.argmax(-1).ravel()
        true = true.ravel()

        # True array may include negative or non-negative integers larger than n_classes, e.g. int class 5 "UNKNOWN"
        # Here we mask out any out-of-range values any evaluate only on in-range classes.
        mask = np.where(np.logical_and(
            np.greater_equal(true, 0),
            np.less(true, self.n_classes)
        ), np.ones_like(true), np.zeros_like(true)).astype(bool)
        pred = pred[mask]
        true = true[mask]

        # Compute relevant CM elements
        # We select the number following the largest class integer when
        # y != pred, then bincount and remove the added dummy class
        tps = np.bincount(np.where(true == pred, true, self.n_classes),
                          minlength=self.n_classes+1)[:-1].astype(np.uint64)
        rel = np.bincount(true, minlength=self.n_classes).astype(np.uint64)
        sel = np.bincount(pred, minlength=self.n_classes).astype(np.uint64)
        return tps, rel, sel

    def predict(self):
        # Get tensors to run and their names
        metrics = getattr(self.model, "loss_functions", self.model.losses) or self.model.loss + self.model.metrics
        metrics = list(filter(lambda m: not type(m) is tf.keras.metrics.Mean, metrics))
        metrics_names = self.model.metrics_names
        self.model.reset_metrics()
        assert len(metrics_names) == len(metrics)

        # Prepare arrays for CM summary stats
        true_pos, relevant, selected, metrics_results = {}, {}, {}, {}
        for id_, sequence in zip(self.IDs, self.sequences):
            # Add count arrays to the result dictionaries
            true_pos[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)
            relevant[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)
            selected[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)

            # Get validation sleep study loader
            n_val = min(len(sequence.dataset_queue), self.max_studies)
            study_iterator = sequence.dataset_queue.get_study_iterator(n_val)

            # Predict and evaluate on all studies
            per_study_metrics = defaultdict(list)
            for i, sleep_study_context in enumerate(study_iterator):
                s = "   {}Validation subject: {}/{}".format(f"[{id_}] "
                                                            if id_ else "",
                                                            i+1,
                                                            n_val)
                print(s, end="\r", flush=True)

                with sleep_study_context as ss:
                    x, y = sequence.get_single_study_full_seq(ss.identifier, reshape=True)
                    pred = self.model.predict_on_batch(x)

                # Compute counts
                if hasattr(pred, "numpy"):
                    pred_numpy = pred.numpy()
                else:
                    pred_numpy = pred
                tps, rel, sel = self._compute_counts(pred=pred_numpy, true=y)
                true_pos[id_] += tps
                relevant[id_] += rel
                selected[id_] += sel

                # Run all metrics
                for metric, name in zip(metrics, metrics_names):
                    res = tf.reduce_mean(metric(y, pred))
                    if hasattr(pred, "numpy"):
                        res = res.numpy()
                    per_study_metrics[name].append(res)
                    if getattr(metric, "stateful", False):
                        if hasattr(metric, "reset_states"):
                            metric.reset_states()
                        else:
                            metric.reset_state()

            # Compute mean metrics for the dataset
            metrics_results[id_] = {}
            for metric, name in zip(metrics, metrics_names):
                metrics_results[id_][name] = np.mean(per_study_metrics[name])
            self.model.reset_metrics()
        return true_pos, relevant, selected, metrics_results

    @staticmethod
    def _compute_dice(tp, rel, sel):
        # Get data masks (to avoid div. by zero warnings)
        # We set precision, recall, dice to 0 in for those particular cls.
        sel_mask = sel > 0
        rel_mask = rel > 0

        # prepare arrays
        precisions = np.zeros(shape=tp.shape, dtype=np.float32)
        recalls = np.zeros_like(precisions)
        dices = np.zeros_like(precisions)

        # Compute precisions, recalls
        precisions[sel_mask] = tp[sel_mask] / sel[sel_mask]
        recalls[rel_mask] = tp[rel_mask] / rel[rel_mask]

        # Compute dice
        intrs = (2 * precisions * recalls)
        union = (precisions + recalls)
        dice_mask = union > 0
        dices[dice_mask] = intrs[dice_mask] / union[dice_mask]

        return precisions, recalls, dices

    def _log_val_results(self, precisions, recalls, dices, metrics, epoch,
                         name, classes):
        # Log the results
        # We add them to a pd dataframe just for the pretty print output
        index = ["cls %i" % i for i in classes]
        metric_keys, metric_vals = map(list, list(zip(*metrics.items())))
        col_order = metric_keys + ["precision", "recall", "dice"]
        nan_arr = np.empty(shape=len(precisions))
        nan_arr[:] = np.nan
        value_dict = {"precision": precisions,
                      "recall": recalls,
                      "dice": dices}
        value_dict.update({key: nan_arr for key in metrics})
        val_results = pd.DataFrame(value_dict,
                                   index=index).loc[:, col_order]  # ensure order
        # Transpose the results to have metrics in rows
        val_results = val_results.T
        # Add mean and set in first row
        means = metric_vals + [precisions.mean(), recalls.mean(), dices.mean()]
        val_results["mean"] = means
        cols = list(val_results.columns)
        cols.insert(0, cols.pop(cols.index('mean')))
        val_results = val_results.loc[:, cols]

        # Print the df to screen
        print_string = val_results.round(self.print_round).to_string()
        logger.info("\n\n" +
                    highlighted(f"[{name}] Validation Results for Epoch {epoch}").lstrip(" ") +
                    "\n" +
                    print_string.replace("NaN", "---"))

    def on_epoch_end(self, epoch, logs=None):
        # Predict and get CM
        TPs, relevant, selected, metrics = self.predict()
        for id_ in self.IDs:
            tp, rel, sel = TPs[id_], relevant[id_], selected[id_]
            precisions, recalls, dices = self._compute_dice(tp=tp, sel=sel, rel=rel)
            classes = np.arange(len(dices))

            # Add to log
            n = (id_ + "_") if len(self.IDs) > 1 else ""
            logs[f"{n}val_dice"] = dices.mean().round(self.log_round)
            logs[f"{n}val_precision"] = precisions.mean().round(self.log_round)
            logs[f"{n}val_recall"] = recalls.mean().round(self.log_round)
            for m_name, value in metrics[id_].items():
                logs[f"{n}val_{m_name}"] = value.round(self.log_round)

            self._log_val_results(precisions=precisions,
                                  recalls=recalls,
                                  dices=dices,
                                  metrics=metrics[id_],
                                  epoch=epoch,
                                  name=id_,
                                  classes=classes)

        if len(self.IDs) > 1:
            # Print cross-dataset mean values
            logger.info(highlighted(f"[ALL DATASETS] Means Across Classes for Epoch {epoch}"))
            fetch = ("val_dice", "val_precision", "val_recall")
            m_fetch = tuple(["val_" + s for s in self.model.metrics_names])
            to_print = {}
            for f in fetch + m_fetch:
                scores = [logs["%s_%s" % (name, f)] for name in self.IDs]
                res = np.mean(scores)
                logs[f] = res.round(self.log_round)  # Add to log file
                to_print[f.split("_")[-1]] = list(scores) + [res]
            df = pd.DataFrame(to_print)
            df.index = self.IDs + ["mean"]
            logger.info("\n" + str(df.round(self.print_round)) + "\n")


class MemoryConsumption(Callback):
    def __init__(self, max_gib=None, round_=2, set_limit=False):
        super().__init__()
        self.max_gib = max_gib
        self.round_ = round_
        if set_limit:
            import resource
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS,
                               (self._gib_to_bytes(max_gib), hard))
            logger.info(f"Setting memory limit to {max_gib} GiB")

    @staticmethod
    def _gib_to_bytes(gib):
        return gib * (1024 ** 3)

    @staticmethod
    def _bytes_to_gib(bytes):
        return bytes / (1024 ** 3)

    def on_epoch_end(self, epoch, logs={}):
        mem_bytes = get_memory_usage()
        mem_gib = round(self._bytes_to_gib(mem_bytes), self.round_)
        logs['memory_usage_gib'] = mem_gib
        if self.max_gib and mem_gib >= self.max_gib:
            logger.warning(f"Stopping training from callback 'MemoryConsumption'! "
                           f"Total memory consumption of {mem_gib} GiB exceeds limitation"
                           f" (self.max_gib = {self.max_gib})")
            self.model.stop_training = True


class MaxTrainingTime(Callback):
    def __init__(self, max_minutes, log_name='train_time_total'):
        """
        TODO
        Args:
        """
        super().__init__()
        self.max_minutes = int(max_minutes)
        self.log_name = log_name

    def on_epoch_end(self, epochs, logs={}):
        """
        TODO

        Args:
            epochs:
            logs:

        Returns:

        """
        train_time_str = logs.get(self.log_name, None)
        if not train_time_str:
            logger.warning(f"Did not find log entry '{self.log_name}' (needed in callback 'MaxTrainingTime')")
            return
        train_time_m = timedelta(
            days=int(train_time_str[:2]),
            hours=int(train_time_str[4:6]),
            minutes=int(train_time_str[8:10]),
            seconds=int(train_time_str[12:14])
        ).total_seconds() / 60
        if train_time_m >= self.max_minutes:
            # Stop training
            logger.warning(f"Stopping training from callback 'MaxTrainingTime'! "
                           f"Total training length of {self.max_minutes} minutes exceeded (now {train_time_m})")
            self.model.stop_training = True


class CarbonUsageTracking(Callback):
    """
    tf.keras Callback for the Carbontracker package.
    See https://github.com/lfwa/carbontracker.
    """
    def __init__(self, epochs, add_to_logs=True, monitor_epochs=-1,
                 epochs_before_pred=-1, devices_by_pid=True, **additional_tracker_kwargs):
        """
        Accepts parameters as per CarbonTracker.__init__
        Sets other default values for key parameters.

        Args:
            add_to_logs: bool, Add total_energy_kwh and total_co2_g to the keras logs after each epoch
            For other arguments, please refer to CarbonTracker.__init__
        """
        super().__init__()
        self.tracker = None
        self.add_to_logs = bool(add_to_logs)
        self.parameters = {"epochs": epochs,
                           "monitor_epochs": monitor_epochs,
                           "epochs_before_pred": epochs_before_pred,
                           "devices_by_pid": devices_by_pid}
        self.parameters.update(additional_tracker_kwargs)

    def on_train_end(self, logs=None):
        """ Ensure actual consumption is reported """
        self.tracker.stop()

    def on_epoch_begin(self, epoch, logs=None):
        """ Start tracking this epoch """
        if self.tracker is None:
            # At this point all CPUs should be discoverable
            self.tracker = CarbonTracker(**self.parameters)
        self.tracker.epoch_start()

    def on_epoch_end(self, epoch, logs={}):
        """ End tracking this epoch """
        self.tracker.epoch_end()
        if self.add_to_logs:
            energy_kwh = self.tracker.tracker.total_energy_per_epoch().sum()
            co2eq_g = self.tracker._co2eq(energy_kwh)
            logs["total_energy_kwh"] = round(energy_kwh, 6)
            logs["total_co2_g"] = round(co2eq_g, 6)


class LearningCurve(Callback):
    """
    On epoch end this callback looks for all csv files matching the 'csv_regex'
    regex within the dir 'out_dir' and attempts to create a learning curve for
    each file that will be saved to 'out_dir'.

    Note: Failure to plot a learning curve based on a given csv file will
          is handled in the plot_all_training_curves function and will not
          cause the LearningCurve callback to raise an exception.
    """
    def __init__(self, log_dir="logs", out_dir="logs", fname="curve.png",
                 csv_regex="*training.csv", **plot_kwargs):
        """
        Args:
            log_dir: Relative path from the
            out_dir:
            fname:
            csv_regex:
        """
        super().__init__()
        out_dir = os.path.abspath(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.csv_regex = os.path.join(os.path.abspath(log_dir), csv_regex)
        self.save_path = os.path.join(out_dir, fname)
        self.plot_kwargs = plot_kwargs

    def on_epoch_end(self, epoch, logs=None):
        try:
            plot_all_training_curves(self.csv_regex,
                                     self.save_path,
                                     logy=True,
                                     **self.plot_kwargs)
        except Exception as e:
            logger.error(f"Could not plot one or more training curves. Reason: {e}")


class DelayedCallback(object):
    """
    Callback wrapper that delays the functionality of another callback by N
    number of epochs.
    """
    def __init__(self, callback, start_from=0):
        """
        Args:
            callback:   A tf.keras callback
            start_from: Delay the activity of 'callback' until this epoch
                        'start_from'
        """
        self.callback = callback
        self.start_from = start_from

    def __getattr__(self, item):
        return getattr(self.callback, item)

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_from-1:
            self.callback.on_epoch_end(epoch, logs=logs)
        else:
            logger.info(f"[{self.callback.__class__.__name__}] "
                        f"Not active at epoch {epoch+1} - will be at {self.start_from}")


class TrainTimer(Callback):
    """
    Appends train timing information to the log.
    If called prior to tf.keras.callbacks.CSVLogger this information will
    be written to disk.
    """
    def __init__(self, max_minutes=None, verbose=1):
        super().__init__()
        self.max_minutes = int(max_minutes) if max_minutes else None
        self.verbose = bool(verbose)

        # Timing attributes
        self.train_begin_time = None
        self.prev_epoch_time = None

    def on_train_begin(self, logs=None):
        self.train_begin_time = datetime.now()

    def on_epoch_begin(self, epoch, logs=None):
        self.prev_epoch_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        # Compute epoch execution time
        end_time = datetime.now()
        epoch_time = end_time - self.prev_epoch_time
        train_time = end_time - self.train_begin_time

        # Update attributes
        self.prev_epoch_time = end_time

        # Add to logs
        train_hours = round(train_time.total_seconds() / 3600, 4)
        epoch_minutes = round(epoch_time.total_seconds() / 60, 4)
        logs["epoch_minutes"] = epoch_minutes
        logs["train_hours"] = train_hours

        if self.verbose:
            logger.info(f"[TrainTimer] Epoch time: {epoch_minutes:.2f} minutes "
                        f"- Total train time: {train_hours:.2f} hours")
        if self.max_minutes and train_hours*60 > self.max_minutes:
            logger.info(f"Stopping training. Training ran for {train_hours*60} minutes, "
                        f"max_minutes of {self.max_minutes} was specified on the "
                        f"TrainTimer callback.")
            self.model.stop_training = True


class MeanReduceLogArrays(Callback):
    """
    On epoch end, goes through the log and replaces any array entries with
    their mean value.
    """
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if isinstance(value, (np.ndarray, list)):
                logs[key] = np.mean(value)


class PrintDividerLine(Callback):
    """
    Simply prints a line to screen after each epoch
    """
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        print("\n" + "-"*45 + "\n")
