import tensorflow as tf
import tensorlayer as tl

class ResModel(object):

    def __init__(self, num_classes):
        self.blocks_per_group = 4
        self.widening_factor = 4
        self.nb_classes = num_classes

    def reset(self, first):
        self.first = first
        if self.first is True:
            self.sess.close()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)

    def zero_pad_channels(self, x, pad=0):
        """
        Function for Lambda layer
        """
        pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
        return tf.pad(x, pattern)

    def residual_block(self, x, count, nb_filters=16, subsample_factor=1):
        prev_nb_channels = x.outputs.get_shape().as_list()[3]

        if subsample_factor > 1:
            subsample = (1, subsample_factor, subsample_factor, 1)
            # shortcut: subsample + zero-pad channel dim
            name_pool = 'pool_layer' + str(count)
            shortcut = tl.layers.PoolLayer(x,
                                           ksize=subsample,
                                           strides=subsample,
                                           # padding='VALID',
                                           pool=tf.nn.avg_pool,
                                           name=name_pool)

        else:
            subsample = [1, 1, 1, 1]
            # shortcut: identity
            shortcut = x

        if nb_filters > prev_nb_channels:
            name_lambda = 'lambda_layer' + str(count)
            shortcut = tl.layers.LambdaLayer(
                shortcut,
                self.zero_pad_channels,
                fn_args={'pad': nb_filters - prev_nb_channels},
                name=name_lambda)

        name_norm = 'norm' + str(count)
        y = tl.layers.BatchNormLayer(x,
                                     decay=0.999,
                                     epsilon=1e-05,
                                     is_train=True,
                                     name=name_norm)

        name_conv = 'conv_layer' + str(count)
        y = tl.layers.Conv2dLayer(y,
                                  act=tf.nn.relu,
                                  shape=(3, 3, prev_nb_channels, nb_filters),
                                  strides=subsample,
                                  padding='SAME',
                                  name=name_conv)

        name_norm_2 = 'norm_second' + str(count)
        y = tl.layers.BatchNormLayer(y,
                                     decay=0.999,
                                     epsilon=1e-05,
                                     is_train=True,
                                     name=name_norm_2)

        prev_input_channels = y.outputs.get_shape().as_list()[3]
        name_conv_2 = 'conv_layer_second' + str(count)
        y = tl.layers.Conv2dLayer(y,
                                  act=tf.nn.relu,
                                  shape=(3, 3, prev_input_channels, nb_filters),
                                  strides=(1, 1, 1, 1),
                                  padding='SAME',
                                  name=name_conv_2)

        name_merge = 'merge' + str(count)
        out = tl.layers.ElementwiseLayer([y, shortcut],
                                         combine_fn=tf.add,
                                         name=name_merge)


        return out

    def create_model(self, x_batch):

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            x = tl.layers.InputLayer(x_batch, name='input_layer')
            x = tl.layers.Conv2dLayer(x,
                                      act=tf.nn.relu,
                                      shape=(5, 5, 1, 16),
                                      strides=(1, 2, 2, 1),
                                      padding='SAME',
                                      name='cnn_layer_first')

            for i in range(0, self.blocks_per_group):
                nb_filters = 16 * self.widening_factor
                count = i
                x = self.residual_block(x, count, nb_filters=nb_filters, subsample_factor=1)

            for i in range(0, self.blocks_per_group):
                nb_filters = 32 * self.widening_factor
                # if i == 0:
                #     subsample_factor = 2
                # else:
                subsample_factor = 1
                count = i + self.blocks_per_group
                x = self.residual_block(x, count, nb_filters=nb_filters, subsample_factor=subsample_factor)

            for i in range(0, self.blocks_per_group):
                nb_filters = 32 * self.widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                count = i + 2*self.blocks_per_group
                x = self.residual_block(x, count, nb_filters=nb_filters, subsample_factor=subsample_factor)

            x = tl.layers.BatchNormLayer(x,
                                         decay=0.999,
                                         epsilon=1e-05,
                                         is_train=True,
                                         name='norm_last')

            x = tl.layers.PoolLayer(x,
                                    ksize=(1, 5, 5, 1),
                                    strides=(1, 5, 5, 1),
                                    padding='SAME',
                                    pool=tf.nn.avg_pool,
                                    name='pool_last')

            x = tl.layers.FlattenLayer(x, name='flatten')

            x = tl.layers.DenseLayer(x,
                                     n_units=self.nb_classes,
                                     act=tf.identity,
                                     name='fc')

        return x


