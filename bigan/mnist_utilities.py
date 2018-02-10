import tensorflow as tf

"""MNIST BiGAN architecture.

Generator (decoder), encoder and discriminator.

"""

learning_rate = 0.00001
batch_size = 100
layer = 1
latent_dim = 200
dis_inter_layer_dim = 1024
init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)

def encoder(x_inp, is_training=False, getter=None, reuse=False):
    """ Encoder architecture in tensorflow

    Maps the data into the latent space

    Note:
        Provides histogram and distribution tensorflow summaries

    Args:
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not

    Returns:
        (tensor): last activation layer of the encoder

    """
    with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):

        x_inp = tf.reshape(x_inp, [-1, 28, 28, 1])

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d(x_inp,
                                   32,
                                   [3, 3],
                                   padding='SAME',
                                   kernel_initializer=init_kernel,
                                   name='conv')
            make_histogram_summary(net, init_layer=True)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d(net,
                                   64,
                                   [3, 3],
                                   padding='SAME',
                                   strides=[2, 2],
                                   kernel_initializer=init_kernel,
                                   name='conv')
            net = tf.layers.batch_normalization(net,
                                                training=is_training)
            net = leakyReLu(net, name='leaky_relu')
            make_histogram_summary(net)

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d(net,
                                   128,
                                   [3, 3],
                                   padding='SAME',
                                   strides=[2, 2],
                                   kernel_initializer=init_kernel,
                                   name='conv')
            net = tf.layers.batch_normalization(net,
                                                training=is_training)
            net = leakyReLu(net, name='leaky_relu')
            make_histogram_summary(net)

        net = tf.contrib.layers.flatten(net)

        name_net = 'layer_4'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=latent_dim,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            make_histogram_summary(net)

        enc_sum = tf.summary.merge_all('encoder')

    return net

def decoder(z_inp, is_training=False, getter=None, reuse=False):
    """ Decoder architecture in tensorflow

    Generates data from the latent space

    Note:
        Provides histogram and distribution tensorflow summaries

    Args:
        z_inp (tensor): variable in the latent space
        reuse (bool): sharing variables or not

    Returns:
        (tensor): last activation layer of the generator

    """
    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(z_inp,
                                  units=1024,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.layers.batch_normalization(net,
                                        training=is_training,
                                        name='batch_normalization')
            net = tf.nn.relu(net, name='relu')
            make_histogram_summary(net, init_layer=True)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=7*7*128,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.layers.batch_normalization(net,
                                        training=is_training,
                                        name='batch_normalization')
            net = tf.nn.relu(net, name='relu')
            make_histogram_summary(net)

        net = tf.reshape(net, [-1, 7, 7, 128])

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                     filters=64,
                                     kernel_size=4,
                                     strides= 2,
                                     padding='same',
                                     kernel_initializer=init_kernel,
                                     name='conv')
            net = tf.layers.batch_normalization(net,
                                        training=is_training,
                                        name='batch_normalization')
            net = tf.nn.relu(net, name='relu')
            make_histogram_summary(net)

        name_net = 'layer_4'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                     filters=1,
                                     kernel_size=4,
                                     strides=2,
                                     padding='same',
                                     kernel_initializer=init_kernel,
                                     name='conv')
            net = tf.tanh(net, name='tanh')
            make_histogram_summary(net)

        gen_sum = tf.summary.merge_all('generator')

    return net

def discriminator(z_inp, x_inp, is_training=False, getter=None, reuse=False):
    """ Discriminator architecture in tensorflow

    Discriminates between pairs (E(x), x) and (z, G(z))

    Note:
        Provides histogram and distribution tensorflow summaries

    Args:
        z_inp (tensor): variable in the latent space
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not

    Returns:
        logits (tensor): last activation layer of the discriminator
        intermediate_layer (tensor): intermediate layer for feature matching

    """
    with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):

        # D(x)
        x_inp = tf.reshape(x_inp, [-1, 28, 28, 1])

        name_net = 'x_layer_1'
        with tf.variable_scope(name_net):
            x = tf.layers.conv2d(x_inp,
                           filters=64,
                           kernel_size=4,
                           strides=2,
                           padding='same',
                           kernel_initializer=init_kernel,
                           name='conv')
            x = leakyReLu(x, 0.1, name='leaky_relu')
            x = tf.layers.dropout(x, rate=0.5, name='dropout')
            make_histogram_summary(x, init_layer=True)

        name_net = 'x_layer_2'
        with tf.variable_scope(name_net):
            x = tf.layers.conv2d(x,
                           filters=64,
                           kernel_size=4,
                           strides=2,
                           padding='same',
                           kernel_initializer=init_kernel,
                           name='conv')
            x = tf.layers.batch_normalization(x,
                                        training=is_training,
                                        name='batch_normalization')
            x = leakyReLu(x, 0.1, name='leaky_relu')
            x = tf.layers.dropout(x, rate=0.5, name='dropout')
            make_histogram_summary(x)

        x = tf.reshape(x, [-1, 7 * 7 * 64])

        # D(z)
        name_z = 'z_layer_1'
        with tf.variable_scope(name_z):
            z = tf.layers.dense(z_inp,
                                512,
                                kernel_initializer=init_kernel,
                                name='fc')
            z = leakyReLu(z, name='leaky_relu')
            z = tf.layers.dropout(z, rate=0.5, name='dropout')
            make_histogram_summary(z)

        # D(x,z)
        y = tf.concat([x, z], axis=1)

        name_y = 'y_layer_1'
        with tf.variable_scope(name_y):
            y = tf.layers.dense(y,
                                dis_inter_layer_dim,
                                kernel_initializer=init_kernel,
                                name='fc')
            y = leakyReLu(y, name='leaky_relu')
            y = tf.layers.dropout(y, rate=0.5, name='dropout')
            make_histogram_summary(y)

        intermediate_layer = y

        name_y = 'y_fc_logits'
        with tf.variable_scope(name_y):
            logits = tf.layers.dense(y,
                                     1,
                                     kernel_initializer=init_kernel,
                                     name='fc')
            make_histogram_summary(logits)

        dis_sum = tf.summary.merge_all('discriminator')

    return logits, intermediate_layer

def make_histogram_summary(net, init_layer=False):
    """ Does the histogram summaries in tensorboard"""
    global layer
    if init_layer:
        layer = 1
    scope = tf.get_variable_scope().name
    name_model = scope.split('/')[0]
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    tf.summary.histogram('{}/kernel_{}'.format(scope, layer),
                         net_vars[0], [name_model])
    tf.summary.histogram('{}/bias_{}'.format(scope, layer),
                         net_vars[1], [name_model])
    tf.summary.histogram('{}/act_{}'.format(scope, layer),
                         net, [name_model])
    layer += 1

def leakyReLu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

