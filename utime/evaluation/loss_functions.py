import tensorflow as tf


def _get_shapes_and_one_hot(y_true, y_pred):
    shape = y_pred.get_shape()
    n_classes = shape[-1]
    # Squeeze dim -1 if it is == 1, otherwise leave it
    dims = tf.cond(tf.equal(y_true.shape[-1] or -1, 1), lambda: tf.shape(y_true)[:-1], lambda: tf.shape(y_true))
    y_true = tf.reshape(y_true, dims)
    y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=n_classes)
    return y_true, shape, n_classes


def sparse_dice_loss(y_true, y_pred, smooth=1):
    """
    Approximates the class-wise dice coefficient computed per-batch element
    across spatial image dimensions. Returns the 1 - mean(per_class_dice) for
    each batch element.
    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    intersection = tf.reduce_sum(y_true * y_pred, axis=reduction_dims)
    union = tf.reduce_sum(y_true + y_pred, axis=reduction_dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice, axis=-1, keepdims=True)


class SparseDiceLoss(tf.keras.losses.Loss):
    """ tf reduction wrapper for sparse_dice_loss """
    def __init__(self,
                 reduction,
                 smooth=1,
                 name='sparse_dice_loss',
                 **kwargs):
        self.smooth = smooth
        super(SparseDiceLoss, self).__init__(
            name=name,
            reduction=reduction
        )

    def get_config(self):
        config = super().get_config()
        config.update({'smooth': self.smooth})
        return config

    def call(self, y_true, y_pred):
        return sparse_dice_loss(y_true, y_pred, smooth=self.smooth)
