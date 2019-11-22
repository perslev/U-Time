import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from MultiPlanarUNet.utils import highlighted
from MultiPlanarUNet.logging import ScreenLogger

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from queue import Queue
from threading import Thread


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
    def __init__(self, val_sequence, steps, logger=None, verbose=True):
        """
        Args:
            val_sequence: A deepsleep ValidationMultiSequence object
            steps:        Numer of batches to sample from val_sequences in each
                          validation epoch for each validation set
            logger:       An instance of a MultiPlanar Logger that prints to
                          screen and/or file
            verbose:      Print progress to screen - OBS does not use Logger
        """
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.sequences = val_sequence.sequences
        self.steps = steps
        self.verbose = verbose
        self.n_classes = val_sequence.n_classes
        self.IDs = val_sequence.IDs
        self.print_round = 3
        self.log_round = 4

    def predict(self):
        def eval(queue, steps, TPs, relevant, selected, id_, lock):
            step = 0
            while step < steps:
                # Get prediction and true labels from prediction queue
                step += 1
                p, y = queue.get(block=True)

                # Argmax and CM elements
                p = p.argmax(-1).ravel()
                y = y.ravel()

                # Compute relevant CM elements
                # We select the number following the largest class integer when
                # y != pred, then bincount and remove the added dummy class
                tps = np.bincount(np.where(y == p, y, self.n_classes),
                                  minlength=self.n_classes+1)[:-1]
                rel = np.bincount(y, minlength=self.n_classes)
                sel = np.bincount(p, minlength=self.n_classes)

                # Update counts on shared lists
                lock.acquire()
                TPs[id_] += tps.astype(np.uint64)
                relevant[id_] += rel.astype(np.uint64)
                selected[id_] += sel.astype(np.uint64)
                lock.release()

        # Get tensors to run and their names
        metrics_tensors = self.model.metrics_tensors
        metrics_names = self.model.metrics_names
        assert "loss" in metrics_names and metrics_names.index("loss") == 0
        assert len(metrics_names)-1 == len(metrics_tensors)
        tensors = [self.model.total_loss] + metrics_tensors + self.model.outputs

        # Prepare arrays for CM summary stats
        TPs, relevant, selected, metrics_means = {}, {}, {}, {}
        count_threads = []
        for id_, sequence in zip(self.IDs, self.sequences):
            # Add count arrays to the result dictionaries
            TPs[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)
            relevant[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)
            selected[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)

            # Prepare metrics dicts
            metrics_sums = {
                **{name: 0.0 for name in self.model.metrics_names}
            }

            # Fetch validation samples from the generator
            pool = ThreadPoolExecutor(max_workers=7)
            result = pool.map(sequence.__getitem__, np.arange(self.steps))

            # Prepare queue and thread for computing counts
            count_queue = Queue(maxsize=self.steps)
            count_thread = Thread(target=eval, args=[count_queue, self.steps,
                                                     TPs, relevant, selected,
                                                     id_, Lock()])
            count_threads.append(count_thread)
            count_thread.start()

            sess = tf.keras.backend.get_session()
            for i, res in enumerate(result):
                if self.verbose:
                    s = "   {}Validation step: {}/{}".format(f"[{id_}] "
                                                             if id_ else "",
                                                             i+1, self.steps)
                    print(s, end="\r", flush=True)
                X, y = res
                ins = {
                    self.model.inputs[0]: X,
                    self.model.targets[0]: y
                }
                # Run the specified tensors
                outs = sess.run(tensors, feed_dict=ins)
                loss = outs[0]
                metrics = outs[1:1+len(metrics_tensors)]
                pred = outs[-1]

                # Put values in the queue for counting
                count_queue.put([pred, y])

                # Update metric sums
                for name, value in zip(metrics_names, [loss]+metrics):
                    metrics_sums[name] += value

                # Compute mean metrics
                metrics_means[id_] = {name: value/self.steps for
                                      name, value in metrics_sums.items()}
            pool.shutdown()
            self.logger("")
        self.logger("")
        # Terminate count threads
        for thread in count_threads:
            thread.join()
        return TPs, relevant, selected, metrics_means

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

    def _print_val_results(self, precisions, recalls, dices, epoch,
                           name, classes, logger):
        # Log the results
        # We add them to a pd dataframe just for the pretty print output
        index = ["cls %i" % i for i in classes]
        val_results = pd.DataFrame({
            "precision": precisions,
            "recall": recalls,
            "dice": dices,
        }, index=index)
        # Transpose the results to have metrics in rows
        val_results = val_results.T
        # Add mean and set in first row
        means = [precisions.mean(), recalls.mean(), dices.mean()]
        val_results["mean"] = means
        cols = list(val_results.columns)
        cols.insert(0, cols.pop(cols.index('mean')))
        val_results = val_results.ix[:, cols]

        # Print the df to screen
        logger(highlighted(("[%s] Validation Results for "
                            "Epoch %i" % (name, epoch)).lstrip(" ")))
        logger(val_results.round(self.print_round).to_string() + "\n")

        # Print dataset metrics
        # logger("\nMetrics (batch-wise means, all classes)")
        # longest = max(map(len, metrics.keys()))
        # for m_name, value in metrics.items():
        #     logger(("%s: " % m_name).rjust(longest + 2), round(value, 2))

    def on_epoch_end(self, epoch, logs={}):
        self.logger("\n")
        # Predict and get CM
        TPs, relevant, selected, metrics = self.predict()
        for name in self.IDs:
            tp, rel, sel = TPs[name], relevant[name], selected[name]
            precisions, recalls, dices = self._compute_dice(tp=tp, sel=sel, rel=rel)
            classes = np.arange(len(dices))

            # Add to log
            n = (name + "_") if len(self.IDs) > 1 else ""
            logs[f"{n}val_dice"] = dices.mean().round(self.log_round)
            logs[f"{n}val_precision"] = precisions.mean().round(self.log_round)
            logs[f"{n}val_recall"] = recalls.mean().round(self.log_round)
            for m_name, value in metrics[name].items():
                logs[f"{n}val_{m_name}"] = value.round(self.log_round)

            if self.verbose:
                self._print_val_results(precisions=precisions,
                                        recalls=recalls,
                                        dices=dices,
                                        epoch=epoch,
                                        name=name,
                                        classes=classes,
                                        logger=self.logger)

        if len(self.IDs) > 1:
            # Print cross-dataset mean values
            if self.verbose:
                self.logger(highlighted(f"[ALL DATASETS] Means Across Classes"
                                        f" for Epoch {epoch}"))
            fetch = ("val_dice", "val_precision", "val_recall")
            m_fetch = tuple(["val_" + s for s in self.model.metrics_names])
            to_print = {}
            for f in fetch + m_fetch:
                scores = [logs["%s_%s" % (name, f)] for name in self.IDs]
                res = np.mean(scores)
                logs[f] = res.round(self.log_round)  # Add to log file
                to_print[f.split("_")[-1]] = list(scores) + [res]
            if self.verbose:
                df = pd.DataFrame(to_print)
                df.index = self.IDs + ["mean"]
                print(df.round(self.print_round))
            self.logger("")
