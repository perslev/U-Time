import logging
import numpy as np

logger = logging.getLogger(__name__)


def expand_to_dim(values, dim):
    expanded = []
    for v in values:
        if not isinstance(v, (tuple, list, np.ndarray)):
            v = [v]
        if isinstance(v, np.ndarray):
            v = list(v)
        if len(v) != dim:
            v *= dim
        expanded.append(np.array(v))
    return expanded


def output_features(in_filter_size, padding, kernel_size, stride=1, dim=2):
    in_filter_size, padding, kernel_size, stride = expand_to_dim([in_filter_size,
                                                                  padding,
                                                                  kernel_size,
                                                                  stride],
                                                                 dim=dim)

    return np.floor((in_filter_size + (2*padding) - kernel_size)/stride).astype(np.int) + 1


def output_feature_distance(input_feature_distance, stride, dim=2):
    input_feature_distance, stride = expand_to_dim([input_feature_distance,
                                                    stride], dim=dim)
    return input_feature_distance * stride


def output_receptive_field(input_receptive_field, kernel_size,
                           input_feature_distacne, dim=2):
    input_receptive_field, kernel_size, \
    input_feature_distacne = expand_to_dim([input_receptive_field,
                                            kernel_size,
                                            input_feature_distacne],
                                           dim=dim)
    return input_receptive_field + (kernel_size - 1) * input_feature_distacne


def output_first_feature_center(input_first_feature_center, kernel_size,
                                padding, input_feature_distance, dim=2):
    input_first_feature_center, \
    kernel_size, \
    padding, \
    input_feature_distance = expand_to_dim([input_first_feature_center,
                                            kernel_size, padding,
                                            input_feature_distance],
                                           dim=dim)

    return input_first_feature_center + ((kernel_size-1)/2 - padding) * input_feature_distance


def compute_receptive_fields(layers, verbose=False):
    input_ = layers[0]
    layers = layers[1:]
    size = input_.input.get_shape().as_list()[1:-1]
    dim = len(size)

    # Set first layer parameters
    receptive_field = 1
    jump = 1

    # Loop over all layers
    values = []
    for i, layer in enumerate(layers):
        try:
            kernel_size = layer.kernel_size
        except AttributeError as e:
            try:
                # Pooling layer?
                kernel_size = layer.pool_size
            except AttributeError:
                # Batch norm, flatten etc.
                continue
        kernel_size = np.array(kernel_size)

        # Get potential dilation rates
        try:
            dilation = np.array(layer.dilation_rate).astype(np.int)
        except AttributeError:
            dilation = np.ones(shape=[dim], dtype=np.int)
        if hasattr(layer, "dilations"):
            assert (dilation == 1).all()
            dilation = np.array(layer.dilations)
            dilation = dilation[dilation[:, 0].argmax()]

        # Get strides
        stride = layer.strides

        # Get kernel size taking into account dilation rate
        ks = kernel_size * dilation
        m = np.where(dilation > 1)
        ks[m] -= (dilation[m]-1)

        size = np.array(layer.output.get_shape().as_list()[1:-1])
        jump = output_feature_distance(jump, stride, dim)
        receptive_field = output_receptive_field(receptive_field, ks, jump, dim)

        # Add to list
        values.append((size, jump, receptive_field))

        if verbose:
            s = "\nLayer %i %s(kernel_size=%s, stride=%s, dilation=%s)" % \
                (i+1, layer.__class__.__name__, kernel_size,
                 stride, tuple(dilation))
            logger.info(f"{s}\n"
                        f"{'-' * (len(s) - 1)}\n"
                        "Num feature:".ljust(25) + f"{size}\n" +
                        "Feature distance:".ljust(25) + f"{jump}\n" +
                        "Receptive field:".ljust(25) + f"{receptive_field}"
                        )

    return values
