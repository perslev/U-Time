"""
Implementation of UTime as described in:

Mathias Perslev, Michael Hejselbak Jensen, Sune Darkner, Poul Jørgen Jennum
and Christian Igel. U-Time: A Fully Convolutional Network for Time Series
Segmentation Applied to Sleep Staging. Advances in Neural Information
Processing Systems (NeurIPS 2019)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, MaxPooling2D, Dense, \
                                    UpSampling2D, ZeroPadding2D, Lambda, Conv2D, \
                                    AveragePooling2D, DepthwiseConv2D
from MultiPlanarUNet.logging import ScreenLogger
from MultiPlanarUNet.utils.conv_arithmetics import compute_receptive_fields


class UTime(Model):
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
                 depth=4,
                 dilation=2,
                 activation="elu",
                 dense_classifier_activation="tanh",
                 kernel_size=5,
                 transition_window=1,
                 padding="same",
                 complexity_factor=2,
                 l2_reg=None,
                 pools=(10, 8, 6, 4),
                 data_per_prediction=None,
                 logger=None,
                 build=True,
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
        pools (int or list of ints):
            TODO
        data_per_prediction (int):
            TODO
        logger (MultiPlanarUNet.logging.Logger | ScreenLogger):
            MutliViewUNet.Logger object, logging to files or screen.
        build (bool):
            TODO
        """
        super(UTime, self).__init__()

        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()

        # Set various attributes
        assert len(batch_shape) == 4
        self.n_periods = batch_shape[1]
        self.input_dims = batch_shape[2]
        self.n_channels = batch_shape[3]
        self.n_classes = int(n_classes)
        self.dilation = int(dilation)
        self.cf = complexity_factor
        self.init_filters = int(8 * self.cf)
        self.kernel_size = int(kernel_size)
        self.transition_window = transition_window
        self.activation = activation
        self.l2_reg = l2_reg
        self.depth = depth
        self.n_crops = 0
        self.pools = [pools] * self.depth if not \
            isinstance(pools, (list, tuple)) else pools
        if len(self.pools) != self.depth:
            raise ValueError("Argument 'pools' must be a single integer or a "
                             "list of values of length equal to 'depth'.")
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

        if build:
            # Build model and init base keras Model class
            super().__init__(*self.init_model())

            # Compute receptive field
            ind = [x.__class__.__name__ for x in self.layers].index("UpSampling2D")
            self.receptive_field = compute_receptive_fields(self.layers[:ind])[-1][-1]

            # Log the model definition
            self.log()
        else:
            self.receptive_field = [None]

    @staticmethod
    def create_encoder(in_,
                       depth,
                       pools,
                       filters,
                       kernel_size,
                       activation,
                       dilation,
                       padding,
                       kernel_reg=None,
                       name="encoder",
                       name_prefix=""):
        name = "{}{}".format(name_prefix, name)
        residual_connections = []
        for i in range(depth):
            l_name = name + "_L%i" % i
            conv = Conv2D(filters, (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=kernel_reg,
                          dilation_rate=dilation,
                          name=l_name + "_conv1")(in_)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)
            conv = Conv2D(filters, (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=kernel_reg,
                          dilation_rate=dilation,
                          name=l_name + "_conv2")(bn)
            bn = BatchNormalization(name=l_name + "_BN2")(conv)
            in_ = MaxPooling2D(pool_size=(pools[i], 1),
                               name=l_name + "_pool")(bn)

            # add bn layer to list for residual conn.
            residual_connections.append(bn)
            filters = int(filters * 2)

        # Bottom
        name = "{}bottom".format(name_prefix)
        conv = Conv2D(filters, (kernel_size, 1),
                      activation=activation, padding=padding,
                      kernel_regularizer=kernel_reg,
                      dilation_rate=1,
                      name=name + "_conv1")(in_)
        bn = BatchNormalization(name=name + "_BN1")(conv)
        conv = Conv2D(filters, (kernel_size, 1),
                      activation=activation, padding=padding,
                      kernel_regularizer=kernel_reg,
                      dilation_rate=1,
                      name=name + "_conv2")(bn)
        encoded = BatchNormalization(name=name + "_BN2")(conv)

        return encoded, residual_connections, filters

    def create_upsample(self,
                        in_,
                        res_conns,
                        depth,
                        pools,
                        filters,
                        kernel_size,
                        activation,
                        dilation,  # NOT USED
                        padding,
                        kernel_reg=None,
                        name="upsample",
                        name_prefix=""):
        name = "{}{}".format(name_prefix, name)
        residual_connections = res_conns[::-1]
        for i in range(depth):
            filters = int(filters/2)
            l_name = name + "_L%i" % i

            # Up-sampling block
            fs = pools[::-1][i]
            up = UpSampling2D(size=(fs, 1),
                              name=l_name + "_up")(in_)
            conv = Conv2D(filters, (fs, 1),
                          activation=activation,
                          padding=padding, kernel_regularizer=kernel_reg,
                          name=l_name + "_conv1")(up)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)

            # Crop and concatenate
            cropped_res = self.crop_nodes_to_match(residual_connections[i], bn)
            # cropped_res = residual_connections[i]
            merge = Concatenate(axis=-1,
                                name=l_name + "_concat")([cropped_res, bn])
            conv = Conv2D(filters, (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=kernel_reg,
                          name=l_name + "_conv2")(merge)
            bn = BatchNormalization(name=l_name + "_BN2")(conv)
            conv = Conv2D(filters, (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=kernel_reg,
                          name=l_name + "_conv3")(bn)
            in_ = BatchNormalization(name=l_name + "_BN3")(conv)
        return in_

    def create_dense_modeling(self,
                              in_,
                              in_reshaped,
                              filters,
                              dense_classifier_activation,
                              name_prefix=""):
        cls = Conv2D(filters=filters,
                     kernel_size=(1, 1),
                     activation=dense_classifier_activation,
                     name="{}dense_classifier_out".format(name_prefix))(in_)
        s = (self.n_periods * self.input_dims) - cls.get_shape().as_list()[1]
        out = self.crop_nodes_to_match(
            node1=ZeroPadding2D(padding=[[s // 2, s // 2 + s % 2], [0, 0]])(cls),
            node2=in_reshaped
        )
        return out

    @staticmethod
    def create_seq_modeling(in_,
                            input_dims,
                            data_per_period,
                            n_periods,
                            n_classes,
                            transition_window,
                            activation,
                            name_prefix=""):
        cls = AveragePooling2D((data_per_period, 1),
                               name="{}average_pool".format(name_prefix))(in_)
        out1 = Conv2D(filters=n_classes,
                      kernel_size=(transition_window, 1),
                      activation=activation,
                      padding="same",
                      name="{}sequence_conv_out_1".format(name_prefix))(cls)
        out2 = Conv2D(filters=n_classes,
                      kernel_size=(transition_window, 1),
                      activation="softmax",
                      padding="same",
                      name="{}sequence_conv_out_2".format(name_prefix))(out1)
        s = [-1, n_periods, input_dims//data_per_period, n_classes]
        if s[2] == 1:
            s.pop(2)  # Squeeze the dim
        out = Lambda(lambda x: tf.reshape(x, s),
                     name="{}sequence_classification_reshaped".format(name_prefix))(out2)
        return out

    def init_model(self, inputs=None, name_prefix=""):
        """
        Build the UNet model with the specified input image shape.
        """
        if inputs is None:
            inputs = Input(shape=[self.n_periods,
                                  self.input_dims,
                                  self.n_channels])
        reshaped = [-1, self.n_periods*self.input_dims, 1, self.n_channels]
        in_reshaped = Lambda(lambda x: tf.reshape(x, reshaped))(inputs)

        # Apply regularization if not None or 0
        kr = regularizers.l2(self.l2_reg) if self.l2_reg else None

        settings = {
            "depth": self.depth,
            "pools": self.pools,
            "filters": self.init_filters,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "dilation": self.dilation,
            "padding": self.padding,
            "kernel_reg": kr,
            "name_prefix": name_prefix
        }

        """
        Encoding path
        """
        enc, residual_cons, filters = self.create_encoder(in_=in_reshaped,
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
                                         in_reshaped=in_reshaped,
                                         filters=self.n_classes,
                                         dense_classifier_activation=self.dense_classifier_activation,
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
                                       name_prefix=name_prefix)

        return [inputs], [out]

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping2D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-2]
        s2 = np.array(node2.get_shape().as_list())[1:-2]

        if np.any(s1 != s2):
            self.n_crops += 1
            c = (s1 - s2).astype(np.int)
            cr = np.array([c // 2, c // 2]).flatten()
            cr[self.n_crops % 2] += c % 2
            cropped_node1 = Cropping2D([list(cr), [0, 0]])(node1)
        else:
            cropped_node1 = node1
        return cropped_node1

    def log(self):
        self.logger("{} Model Summary\n"
                    "-------------------".format(__class__.__name__))
        self.logger("N periods:         {}".format(self.n_periods))
        self.logger("Input dims:        {}".format(self.input_dims))
        self.logger("N channels:        {}".format(self.n_channels))
        self.logger("N classes:         {}".format(self.n_classes))
        self.logger("Kernel size:       {}".format(self.kernel_size))
        self.logger("Dilation rate:     {}".format(self.dilation))
        self.logger("CF factor:         %.3f" % self.cf)
        self.logger("Init filters:      {}".format(self.init_filters))
        self.logger("Depth:             %i" % self.depth)
        self.logger("Poolings:          {}".format(self.pools))
        self.logger("Transition window  {}".format(self.transition_window))
        self.logger("Dense activation   {}".format(self.dense_classifier_activation))
        self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Receptive field:   %s" % self.receptive_field[0])
        self.logger("Seq length.:       {}".format(self.n_periods*self.input_dims))
        self.logger("N params:          %i" % self.count_params())
        self.logger("Input:             %s" % self.input)
        self.logger("Output:            %s" % self.output)
