

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
        raise ValueError("Passed 'batch_shape' could not be standardized to"
                         " a length 2 list of format "
                         "[input_dim, input_channels]. Got: {}".format(batch_shape))
