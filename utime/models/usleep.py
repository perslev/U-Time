"""
Implementation of UTime as described in:

Mathias Perslev, Michael Hejselbak Jensen, Sune Darkner, Poul Jørgen Jennum
and Christian Igel. U-Time: A Fully Convolutional Network for Time Series
Segmentation Applied to Sleep Staging. Advances in Neural Information
Processing Systems (NeurIPS 2019)
"""
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, \
                                    Concatenate, MaxPooling2D, \
                                    UpSampling2D, Conv2D, \
                                    AveragePooling2D, Layer
from utime.utils.conv_arithmetics import compute_receptive_fields
from utime.train.utils import get_activation_function

logger = logging.getLogger(__name__)


def shape_safe(input, dim=None):
    if dim is not None:
        return input.shape[dim] or tf.shape(input)[dim]
    else:
        return [shape_safe(input, d) for d in range(len(input.shape))]


class InputReshape(Layer):
    def __init__(self, seq_length, n_channels, name=None, **kwargs):
        super(InputReshape, self).__init__(name=name, **kwargs)
        self.seq_length = seq_length
        self.n_channels = n_channels

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_length": self.seq_length,
            "n_channels": self.n_channels,
        })
        return config

    def call(self, inputs, **kwargs):
        shape = shape_safe(inputs)
        inputs_reshaped = tf.reshape(inputs, shape=[shape[0], self.seq_length or shape[1]*shape[2], 1, self.n_channels])
        return inputs_reshaped


class OutputReshape(Layer):
    def __init__(self, n_periods, name=None, **kwargs):
        super(OutputReshape, self).__init__(name=name, **kwargs)
        self.n_periods = n_periods

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_periods": self.n_periods
        })
        return config

    def call(self, inputs, **kwargs):
        shape = shape_safe(inputs)
        n_pred = int(shape[1] // self.n_periods)
        shape = [shape[0], self.n_periods or shape[1], n_pred, inputs.shape[-1]]
        if n_pred == 1:
            shape.pop(2)
        return tf.reshape(inputs, shape=shape)


class PadEndToEvenLength(Layer):
    def __init__(self, name=None, **kwargs):
        super(PadEndToEvenLength, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        return tf.pad(inputs,
                      paddings=[[0, 0], [0, shape_safe(inputs, 1) % 2], [0, 0], [0, 0]])


class PadToMatch(Layer):
    def __init__(self, name=None, **kwargs):
        super(PadToMatch, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        s = tf.maximum(0, shape_safe(inputs[1], 1) - shape_safe(inputs[0], 1))
        return tf.pad(inputs[0],
                      paddings=[[0, 0], [s // 2, s // 2 + (s % 2)], [0, 0], [0, 0]])


class CropToMatch(Layer):
    def __init__(self, name=None, **kwargs):
        super(CropToMatch, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        diff = tf.maximum(0, shape_safe(inputs[0], 1) - shape_safe(inputs[1], 1))
        start = diff//2 + diff % 2
        return inputs[0][:, start:start+shape_safe(inputs[1], 1), :, :]


class USleep(Model):
    """
    OBS: Uses 2D operations internally with a 'dummy' axis, so that a batch
         of shape [bs, d, c] is processed as [bs, d, 1, c]. These operations
         are (on our systems, at least) currently significantly faster than
         their 1D counterparts in tf.keras.

    See also original U-net paper at http://arxiv.org/abs/1505.04597
    """
    def __init__(self,
                 n_classes,
                 batch_shape,
                 depth=12,
                 dilation=1,
                 activation="elu",
                 dense_classifier_activation="tanh",
                 kernel_size=9,
                 transition_window=1,
                 padding="same",
                 init_filters=5,
                 complexity_factor=2,
                 kernel_initializer=tf.keras.initializers.glorot_uniform(),
                 bias_initializer=tf.keras.initializers.zeros(),
                 l2_reg=None,
                 data_per_prediction=None,
                 no_log=False,
                 name="",
                 **kwargs):
        """
        n_classes (int):
            The number of classes to model, gives the number of filters in the
            final 1x1 conv layer.
        batch_shape (list): Giving the shape of one one batch of data,
                            potentially omitting the zeroth axis (the batch
                            size dim)
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        dilation (int):
            TODO
        activation (string):
            Activation function for convolution layers
        dense_classifier_activation (string):
            TODO
        kernel_size (int):
            Kernel size for convolution layers
        transition_window (int):
            TODO
        padding (string):
            Padding type ('same' or 'valid')
        complexity_factor (int/float):
            Use int(N * sqrt(complexity_factor)) number of filters in each
            convolution layer instead of default N.
        l2_reg (float in [0, 1])
            L2 regularization on conv weights
        data_per_prediction (int):
            TODO
        build (bool):
            TODO
        """
        # Set various attributes
        assert len(batch_shape) == 4
        self.n_periods = batch_shape[1]
        self.input_dims = batch_shape[2]
        self.n_channels = batch_shape[3]
        self.n_classes = int(n_classes)
        self.dilation = int(dilation)
        self.cf = np.sqrt(complexity_factor)
        self.init_filters = init_filters
        self.kernel_size = int(kernel_size)
        self.transition_window = transition_window
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.l2_reg = l2_reg
        self.depth = depth
        self.padding = padding.lower()
        if self.padding != "same":
            raise ValueError("Currently, must use 'same' padding.")

        self.dense_classifier_activation = dense_classifier_activation
        self.data_per_prediction = data_per_prediction or self.input_dims
        if not isinstance(self.data_per_prediction, (int, np.integer)):
            raise TypeError("data_per_prediction must be an integer value")
        if self.input_dims % self.data_per_prediction:
            raise ValueError("'input_dims' ({}) must be evenly divisible by "
                             "'data_per_prediction' ({})".format(self.input_dims,
                                                                 self.data_per_prediction))

        # Build model and init base keras Model class
        super().__init__(*self.init_model(name_prefix=name))

        # Compute receptive field
        ind = [x.__class__.__name__ for x in self.layers].index("UpSampling2D")
        self.receptive_field = compute_receptive_fields(self.layers[:ind])[-1][-1]

        # Log the model definition
        if not no_log:
            self.log()

    @staticmethod
    def create_encoder(in_,
                       depth,
                       filters,
                       kernel_size,
                       activation,
                       dilation,
                       padding,
                       complexity_factor,
                       regularizer=None,
                       name="encoder",
                       name_prefix="",
                       **other_conv_params):
        name = "{}{}".format(name_prefix, name)
        residual_connections = []
        for i in range(depth):
            l_name = name + "_L%i" % i
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          dilation_rate=dilation,
                          name=l_name + "_conv1", **other_conv_params)(in_)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)
            bn = PadEndToEvenLength(name=l_name + "_padding")(bn)
            in_ = MaxPooling2D(pool_size=(2, 1), name=l_name + "_pool")(bn)

            # add bn layer to list for residual conn.
            residual_connections.append(bn)
            filters = int(filters * np.sqrt(2))

        # Bottom
        name = "{}bottom".format(name_prefix)
        conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                      activation=activation, padding=padding,
                      kernel_regularizer=regularizer,
                      bias_regularizer=regularizer,
                      dilation_rate=1,
                      name=name + "_conv1", **other_conv_params)(in_)
        encoded = BatchNormalization(name=name + "_BN1")(conv)
        return encoded, residual_connections, filters

    def create_upsample(self,
                        in_,
                        res_conns,
                        depth,
                        filters,
                        kernel_size,
                        activation,
                        dilation,  # NOT USED
                        padding,
                        complexity_factor,
                        regularizer=None,
                        name="upsample",
                        name_prefix="",
                        **other_conv_params):
        name = "{}{}".format(name_prefix, name)
        residual_connections = res_conns[::-1]
        for i in range(depth):
            filters = int(np.ceil(filters/np.sqrt(2)))
            l_name = name + "_L%i" % i

            # Up-sampling block
            up = UpSampling2D(size=(2, 1), name=l_name + "_up")(in_)
            conv = Conv2D(int(filters*complexity_factor), (2, 1),
                          activation=activation,
                          padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          name=l_name + "_conv1", **other_conv_params)(up)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)

            # Crop and concatenate
            res_con = residual_connections[i]
            cropped_bn = CropToMatch(name=l_name + "_crop")([bn, res_con])
            merge = Concatenate(axis=-1, name=l_name + "_concat")([res_con, cropped_bn])
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          name=l_name + "_conv2", **other_conv_params)(merge)
            in_ = BatchNormalization(name=l_name + "_BN2")(conv)
        return in_

    def create_dense_modeling(self,
                              in_,
                              in_reshaped,
                              filters,
                              dense_classifier_activation,
                              regularizer,
                              complexity_factor,
                              name_prefix="",
                              **other_conv_params):
        cls = Conv2D(filters=int(filters*complexity_factor),
                     kernel_size=(1, 1),
                     kernel_regularizer=regularizer,
                     bias_regularizer=regularizer,
                     activation=dense_classifier_activation,
                     name="{}dense_classifier_out".format(name_prefix),
                     **other_conv_params)(in_)
        cls = PadToMatch(name="{}dense_classifier_out_pad".format(name_prefix))([cls, in_reshaped])
        cls = CropToMatch(name="{}dense_classifier_out_crop".format(name_prefix))([cls, in_reshaped])
        return cls

    @staticmethod
    def create_seq_modeling(in_,
                            input_dims,
                            data_per_period,
                            n_periods,
                            n_classes,
                            transition_window,
                            activation,
                            regularizer=None,
                            name_prefix="",
                            **other_conv_params):
        cls = AveragePooling2D((data_per_period, 1),
                               name="{}average_pool".format(name_prefix))(in_)
        out = Conv2D(filters=n_classes,
                     kernel_size=(transition_window, 1),
                     activation=activation,
                     kernel_regularizer=regularizer,
                     bias_regularizer=regularizer,
                     padding="same",
                     name="{}sequence_conv_out_1".format(name_prefix),
                     **other_conv_params)(cls)
        out = Conv2D(filters=n_classes,
                     kernel_size=(transition_window, 1),
                     activation="softmax",
                     kernel_regularizer=regularizer,
                     bias_regularizer=regularizer,
                     padding="same",
                     name="{}sequence_conv_out_2".format(name_prefix),
                     **other_conv_params)(out)
        out = OutputReshape(n_periods=n_periods, name="{}output_reshape".format(name_prefix))(out)
        return out

    def init_model(self, inputs=None, name_prefix=""):
        """
        Build the UNet model with the specified input image shape.
        """
        seq_length = self.n_periods * self.input_dims if self.n_periods else None
        if inputs is None:
            inputs = Input(shape=[self.n_periods, self.input_dims, self.n_channels])
        inputs_reshaped = InputReshape(seq_length, self.n_channels)(inputs)
        
        # Apply regularization if not None or 0
        regularizer = regularizers.l2(self.l2_reg) if self.l2_reg else None

        # Get activation func from tf or tfa
        activation = get_activation_function(activation_string=self.activation)

        settings = {
            "depth": self.depth,
            "filters": self.init_filters,
            "kernel_size": self.kernel_size,
            "activation": activation,
            "dilation": self.dilation,
            "padding": self.padding,
            "regularizer": regularizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "name_prefix": name_prefix,
            "complexity_factor": self.cf
        }

        """
        Encoding path
        """
        enc, residual_cons, filters = self.create_encoder(in_=inputs_reshaped,
                                                          **settings)

        """
        Decoding path
        """
        settings["filters"] = filters
        up = self.create_upsample(enc, residual_cons, **settings)

        """
        Dense class modeling layers
        """
        cls = self.create_dense_modeling(in_=up,
                                         in_reshaped=inputs_reshaped,
                                         filters=self.n_classes,
                                         dense_classifier_activation=self.dense_classifier_activation,
                                         regularizer=regularizer,
                                         complexity_factor=self.cf,
                                         name_prefix=name_prefix)

        """
        Sequence modeling
        """
        out = self.create_seq_modeling(in_=cls,
                                       input_dims=self.input_dims,
                                       data_per_period=self.data_per_prediction,
                                       n_periods=self.n_periods,
                                       n_classes=self.n_classes,
                                       transition_window=self.transition_window,
                                       activation=self.activation,
                                       regularizer=regularizer,
                                       name_prefix=name_prefix)

        return [inputs], [out]

    def log(self):
        logger.info(f"\nUSleep Model Summary\n"
                    "--------------------\n"
                    f"N periods:         {self.n_periods or 'ANY'}\n"
                    f"Input dims:        {self.input_dims}\n"
                    f"N channels:        {self.n_channels}\n"
                    f"N classes:         {self.n_classes}\n"
                    f"Kernel size:       {self.kernel_size}\n"
                    f"Dilation rate:     {self.dilation}\n"
                    f"CF factor:         {self.cf**2:.3f}\n"
                    f"Init filters:      {self.init_filters}\n"
                    f"Depth:             {self.depth}\n"
                    f"Pool size:         2\n"
                    f"Transition window  {self.transition_window}\n"
                    f"Dense activation   {self.dense_classifier_activation}\n"
                    f"l2 reg:            {self.l2_reg}\n"
                    f"Padding:           {self.padding}\n"
                    f"Conv activation:   {self.activation}\n"
                    f"Receptive field:   {self.receptive_field[0]}\n"
                    f"Seq length.:       {self.n_periods*self.input_dims if self.n_periods else 'ANY'}\n"
                    f"N params:          {self.count_params()}\n"
                    f"Input:             {self.input}\n"
                    f"Output:            {self.output}")
