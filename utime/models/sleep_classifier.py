import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (LSTM, Bidirectional, Input, Lambda,
                                     AveragePooling1D, TimeDistributed,
                                     Activation)
from tensorflow.keras.models import Model


def reshape(layer, shape):
    return tf.reshape(layer, shape)


def aggregator(x, factor=16, hidden_units=10):
    d, c = x.get_shape()[1], x.get_shape()[-1]
    x = Lambda(reshape, arguments={"shape": [-1, int(d//factor), factor, c]})(x)
    x1 = TimeDistributed(Bidirectional(LSTM(units=hidden_units,
                                            return_sequences=True,
                                            recurrent_activation='elu',
                                            activation='elu')))(x)
    x2 = TimeDistributed(Bidirectional(LSTM(units=1,
                                            recurrent_activation='elu',
                                            activation=None)))(x1)
    x_out = Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(x2)
    x_out = Activation("sigmoid")(x_out)
    return x_out


class SleepClassifier(Model):
    def __init__(self,
                 n_input_seconds,
                 sample_rate,
                 reduction_factor,
                 steps,
                 n_input_channels=5,
                 n_hidden_units=10):

        input = Input(shape=[int(sample_rate*n_input_seconds),
                             n_input_channels])
        x = input
        for _ in range(steps):
            x = aggregator(x, reduction_factor, hidden_units=n_hidden_units)
        super().__init__([input], [x])


reduction_factor = 8
sample_rate = 128
n_seconds = 2048

x_in = tf.convert_to_tensor(np.random.randn(1, sample_rate*n_seconds, 5).astype(np.float32))

m = SleepClassifier(n_seconds, sample_rate, reduction_factor, 6)
m.summary()
