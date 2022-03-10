"""
A set of functions for running prediction in various settings
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def predict_on_generator(model, generator, argmax=False):
    """
    Takes a tf.keras model and uses it to predict on all batches in a generator
    Stacks the predictions over all batches on axis 0 (vstack)

    Args:
        model:      A tf.keras module instance. Should accept batches as output
                    from 'generator'
        generator:  A generator object yielding one or more batches of data to
                    predict on
        argmax:     Whether to return argmax values or model output values

    Returns:
        If argmax is true, returns integer predictions of shape [-1, 1].
        Otherwise, returns floating values of shape [-1, n_classes]
    """
    pred = []
    end_of_data = False
    while not end_of_data:
        try:
            X_batch, _ = next(generator)
        except StopIteration:
            end_of_data = True
        else:
            # Predict
            pred_batch = model.predict_on_batch(X_batch)
            if argmax:
                pred_batch = pred_batch.argmax(-1).reshape(-1, 1)
            pred.append(pred_batch)
    return np.vstack(pred)


def predict_by_id(model, sequencer, study_id, argmax=False):
    """
    Takes a tf.keras model and predicts on all batches of data in a SleepStudy
    object.

    Args:
        model:      A tf.keras model instance. Should accept batches of data
                    as output by the 'sequence' Sequence object.
        sequencer:  A Sequence object which stores at least the passed
                    SleepStudy object of 'sleep_study'.
        study_id:   The identifier string of a SleepStudy object in 'sequence'.
        argmax:     See predict_on_generator docstring.

    Returns:
        Predictions of 'model' on all batches of data in a SleepStudy
        Please refer to the 'predict_on_generator' docstring.
    """
    # Get generator
    gen = sequencer.to_batch_generator(study_id=study_id)
    return predict_on_generator(model, gen, argmax)


def sequence_predict_generator(model, total_seq_length, generator,
                               argmax=False, overlapping=True, verbose=True):
    """
    Takes a tf.keras model and predicts on segments of data from a generator.
    This function takes a few additional values needed to derive an
    understanding of the data produced by 'generator', see below:

    Args:
        model:             A tf.keras model to predict with. Should accept data
                           as output by the generator.
        total_seq_length:  The total number of 'segments/epochs/stages' in the
                           generator. This is needed to initialize the
                           predictions array.
        generator:         A generator which produces batches of data
        argmax:            Whether to return argmax values or model output values
        overlapping:       Specifies whether the sequences output of 'generator'
                           represent overlapping segments or contagious data.
        verbose:           If True, prints the prediction progess to screen.

    Returns:
        An array of shape [total_seq_length, n_classes] or
        [total_seq_length, -1, n_classes] if data_per_prediction != input_dims.
        If argmax = True axis -1 (now shape 1) is squeezed.
    """
    n_classes = model.outputs[0].get_shape()[-1]
    s = model.outputs[0].get_shape().as_list()
    pred = np.zeros(shape=[total_seq_length] + s[2:], dtype=np.float64)

    cur_pos = 0
    for X, _, _ in generator:
        if verbose:
            print("  pos: {}/{}".format(cur_pos+1, total_seq_length),
                  end="\r", flush=True)
        batch_pred = model.predict_on_batch(X)
        if overlapping:
            for p in batch_pred:
                pred[cur_pos:cur_pos+p.shape[0]] += p
                cur_pos += 1
        else:
            batch_pred = batch_pred.reshape(-1, n_classes)
            n_vals = batch_pred.shape[0]
            pred[cur_pos:cur_pos+n_vals] += batch_pred
            cur_pos += n_vals
    if argmax:
        pred = pred.argmax(-1)
    print()
    return pred
