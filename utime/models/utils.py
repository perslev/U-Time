import logging
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

logger = logging.getLogger(__name__)


def standardize_batch_shape(batch_shape):
    batch_shape = list(batch_shape)
    if len(batch_shape) == 3:
        return batch_shape[1:]
    elif len(batch_shape) == 2:
        return batch_shape
    elif len(batch_shape) == 1:
        return batch_shape + [1]
    elif len(batch_shape) == 4:
        return batch_shape[2:]
    else:
        raise ValueError(f"Passed 'batch_shape' could not be standardized to"
                         f" a length 2 list of format "
                         f"[input_dim, input_channels]. Got: {batch_shape}")


def save_frozen_model(keras_model, out_folder, pb_file_name):
    assert len(keras_model.inputs) == 1, 'Only implemented for models wit 1 input'
    assert len(keras_model.outputs) == 1, 'Only implemented for models wit 1 output'
    model = tf.function(keras_model, input_signature=[tf.TensorSpec(keras_model.inputs[0].shape,
                                                                    keras_model.inputs[0].dtype)])
    frozen_func = convert_variables_to_constants_v2(model.get_concrete_function())
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=out_folder,
                      name=pb_file_name,
                      as_text=False)
