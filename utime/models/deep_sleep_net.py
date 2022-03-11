"""
Reimplementation of DeepSleepNet as described in
A. Supratak, H. Dong, C. Wu and Y. Guo, "DeepSleepNet: A Model for Automatic
Sleep Stage Scoring Based on Raw Single-Channel EEG," in IEEE Transactions
on Neural Systems and Rehabilitation Engineering, vol. 25, no. 11,
pp. 1998-2008, Nov. 2017. doi: 10.1109/TNSRE.2017.2721116
"""

import logging
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, \
                                    Input, Flatten, Dense, Dropout, Concatenate, \
                                    Lambda, LSTM, Bidirectional, Add
from utime.models.utils import standardize_batch_shape

logger = logging.getLogger(__name__)


class DeepFeatureNet(tf.keras.Model):
    """
    CNN/Representation learning sub-network
    """
    def __init__(self,
                 batch_shape,
                 n_classes,
                 padding="valid",
                 activation="relu",
                 use_dropout=True,
                 use_bn=True,
                 classify=True,
                 flatten=True,
                 l2_reg=0.0,
                 log=True,
                 build=True,
                 **unused):
        super(DeepFeatureNet, self).__init__()
        self.batch_shape = standardize_batch_shape(batch_shape)
        self.n_classes = n_classes
        self.use_dropout = use_dropout
        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        self.classify = classify
        self.flatten = flatten
        self.l2_reg = l2_reg
        self.reg = None
        self.model_name = "DeepFeatureNet"

        # Build model and init base keras Model class
        if build:
            with tf.name_scope(self.model_name):
                super(DeepFeatureNet, self).__init__(*self.init_model())
            if log:
                self.log()

    def log(self):
        logger.info("\nDeepFeatureNet Model Summary\n"
                    "----------------------------\n"
                    f"Batch shape:       {self.batch_shape}\n"
                    f"N classes:         {self.n_classes}\n"
                    f"l2 reg:            {self.l2_reg}\n"
                    f"Padding:           {self.padding}\n"
                    f"Conv activation:   {self.activation}\n"
                    f"N params:          {self.count_params()}\n"
                    f"Input:             {self.input}\n"
                    f"Output:            {self.output}")

    def _build_encoder(self, inputs, n_filters_input, filter_size_input,
                       stride_input, pool_size_input, n_filters_lower,
                       filter_size_lower, stride_lower, pool_size_lower, kr,
                       name):
        inputs = Conv1D(filters=n_filters_input,
                        kernel_size=filter_size_input,
                        strides=stride_input,
                        kernel_regularizer=kr,
                        activation=self.activation,
                        padding=self.padding,
                        name=f"{name}_conv_input")(inputs)
        if self.use_bn:
            inputs = BatchNormalization(name=f"{name}_BN_input")(inputs)
        inputs = MaxPooling1D(pool_size=pool_size_input,
                              name=f"{name}_MP_input")(inputs)
        if self.use_dropout:
            inputs = Dropout(0.5, name=f"{name}_DO_input")(inputs)
        for i in range(3):
            inputs = Conv1D(filters=n_filters_lower,
                            kernel_size=filter_size_lower,
                            strides=stride_lower,
                            kernel_regularizer=kr,
                            activation=self.activation,
                            padding=self.padding,
                            name=f"{name}_conv_lower{i}")(inputs)
            if self.use_bn:
                inputs = BatchNormalization(name=f"{name}BN_lower_{i}")(inputs)
        inputs = MaxPooling1D(pool_size=pool_size_lower,
                              name=f"{name}_MP_lower")(inputs)
        if self.use_dropout:
            inputs = Dropout(0.5, name=f"{name}_DO_lower")(inputs)
        return inputs

    def init_model(self, inputs=None):
        if inputs is None:
            inputs = Input(shape=self.batch_shape, name="input")

        # Apply regularization if not None or 0
        self.reg = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg else None

        # Build two encoders
        with tf.name_scope("small_filter_encoder"):
            enc1 = self._build_encoder(inputs=inputs,
                                       n_filters_input=64,
                                       filter_size_input=50,
                                       stride_input=6,
                                       pool_size_input=8,
                                       n_filters_lower=128,
                                       filter_size_lower=8,
                                       stride_lower=1,
                                       pool_size_lower=4,
                                       kr=self.reg, name="SFE")
        with tf.name_scope("large_filter_encoder"):
            enc2 = self._build_encoder(inputs=inputs,
                                       n_filters_input=64,
                                       filter_size_input=400,
                                       stride_input=50,
                                       pool_size_input=4,
                                       n_filters_lower=128,
                                       filter_size_lower=6,
                                       stride_lower=1,
                                       pool_size_lower=2,
                                       kr=self.reg, name="LFE")
        if self.flatten:
            enc1 = Flatten(name="flatten_small_filters")(enc1)
            enc2 = Flatten(name="flatten_large_filters")(enc2)
        if self.classify:
            outputs = Concatenate(name="concat_features")([enc1, enc2])
            outputs = Dense(self.n_classes, activation="softmax",
                            name="classifier")(outputs)
        else:
            outputs = Concatenate(name="concat")([enc1, enc2])
        return [inputs], [outputs]


class DeepSleepNet(DeepFeatureNet):
    """
    Full DeepSleepNet model (feature + sequence learning)
    """
    def __init__(self,
                 batch_shape,
                 n_classes,
                 n_rnn_layers=2,
                 padding="same",
                 activation="relu",
                 use_dropout=True,
                 use_bn=True,
                 l2_reg=0.0,
                 l2_reg_feature_net=None,
                 log=True,
                 name="DeepSleepNet",
                 **unused):
        super(DeepSleepNet, self).__init__(
            batch_shape=[None, None],
            n_classes=n_classes,
            padding=padding,
            activation=activation,
            use_dropout=use_dropout,
            use_bn=use_bn,
            classify=False,
            l2_reg=l2_reg_feature_net or l2_reg,
            log=False,
            build=False
        )
        assert len(batch_shape) == 4
        self.n_periods = batch_shape[1]
        self.input_dims = batch_shape[2]
        self.n_channels = batch_shape[3]
        self.n_rnn_layers = n_rnn_layers
        self.model_name = name
        with tf.name_scope(self.model_name):
            super(DeepFeatureNet, self).__init__(
                *self.init_model()
            )
        if log:
            self.log()

    def log(self):
        logger.info("DeepSleepNet Model Summary\n"
                    "-------------------------"
                    f"N periods:         {self.n_periods}\n"
                    f"Input dims:        {self.input_dims}\n"
                    f"N classes:         {self.n_classes}\n"
                    f"N RNN layers:      {self.n_rnn_layers}\n"
                    f"l2 reg:            {self.l2_reg}\n"
                    f"Padding:           {self.padding}\n"
                    f"Conv activation:   {self.activation}\n"
                    f"N params:          {self.count_params()}\n"
                    f"Input:             {self.input}\n"
                    f"Output:            {self.output}")

    def _reshape(self, layer, shape):
        return tf.keras.backend.reshape(layer, shape)

    def init_model(self, inputs=None):
        inputs = Input(shape=[self.n_periods, self.input_dims, self.n_channels])

        # Reshape and build DeepFeatureNet
        s = [-1, self.input_dims, self.n_channels]
        feature_ins = Lambda(self._reshape, arguments={"shape": s})(inputs)
        _, features = super(DeepSleepNet, self).init_model(inputs=feature_ins)
        features = features[0]

        outputs = []
        with tf.name_scope("fc_skip_conn"):
            # Fully connected skip-conn
            skip_con = Dense(units=1024, activation="relu",
                             name="skip_conn_FC",
                             kernel_regularizer=self.reg)(features)
            outputs.append(BatchNormalization(name="skip_conn_BN")(skip_con))
        with tf.name_scope("bidirect_LSTMs"):
            # Reshape to sequence and feed to Bidirectional LSTMs
            s = [-1, self.n_periods, features.get_shape()[-1].value]
            seq = Lambda(self._reshape, arguments={"shape": s},
                         name="LSTM_input_reshape")(features)
            do = 0.5 if self.use_dropout else 0
            for i in range(self.n_rnn_layers):
                seq = Bidirectional(LSTM(units=512, dropout=do,
                                         recurrent_dropout=0,
                                         return_sequences=True,
                                         kernel_regularizer=self.reg,
                                         name=f"LSTM_{i}"),
                                    name=f"bidirect_{i}")(seq)
                if self.use_bn:
                    seq = BatchNormalization(name=f"LSTM_bn_{i}")(seq)
            s = [-1, 1024]
            seq = Lambda(self._reshape, arguments={"shape": s},
                         name="LSTM_output_reshape")(seq)
            outputs.append(seq)
        with tf.name_scope("LSTM_and_skip_add"):
            # Add skip and LSTM outputs
            outputs = Add(name="add_LSTM_and_skip")(outputs)
            if self.use_dropout:
                outputs = Dropout(0.5)(outputs)
        with tf.name_scope("classifier"):
            # Classify
            outputs = Dense(units=self.n_classes, activation="softmax",
                            name="deep_sleep_net_classifier")(outputs)
            s = [-1, self.n_periods, self.n_classes]
            outputs = Lambda(self._reshape, arguments={"shape": s},
                             name="output_reshape")(outputs)
        return [inputs], [outputs]
