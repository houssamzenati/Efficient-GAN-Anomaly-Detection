import tensorflow as tf

"""KDD BiGAN architecture.

Generator (decoder), encoder and discriminator.

"""


learning_rate = 0.00001
batch_size = 50
layer = 1
latent_dim = 32
dis_inter_layer_dim = 128
init_kernel = tf.contrib.layers.xavier_initializer()

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

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(x_inp,
                              units=64,
                              kernel_initializer=init_kernel,
                              name='fc')
            net = leakyReLu(net)
            make_histogram_summary(net, init_layer=True)

        name_net = 'layer_2'
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
                                  units=64,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)
            make_histogram_summary(net, init_layer=True)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=128,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)
            make_histogram_summary(net)

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=121,
                                  kernel_initializer=init_kernel,
                                  name='fc')
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
        logits (tensor): last activation layer of the discriminator (shape 1)
        intermediate_layer (tensor): intermediate layer for feature matching

    """
    with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):
        # D(x)
        name_x = 'x_layer_1'
        with tf.variable_scope(name_x):
            x = tf.layers.dense(x_inp,
                          units=128,
                          kernel_initializer=init_kernel,
                          name='fc')
            x = leakyReLu(x)
            x = tf.layers.dropout(x, rate=0.2, name='dropout', training=is_training)
            make_histogram_summary(x, init_layer=True)

        # D(z)
        name_z = 'z_fc_1'
        with tf.variable_scope(name_z):
            z = tf.layers.dense(z_inp, 128, kernel_initializer=init_kernel)
            z = leakyReLu(z)
            z = tf.layers.dropout(z, rate=0.2, name='dropout', training=is_training)
            make_histogram_summary(z)

        # D(x,z)
        y = tf.concat([x, z], axis=1)

        name_y = 'y_fc_1'
        with tf.variable_scope(name_y):
            y = tf.layers.dense(y,
                                dis_inter_layer_dim,
                                kernel_initializer=init_kernel)
            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.2, name='dropout', training=is_training)
            make_histogram_summary(y)

        intermediate_layer = y

        name_y = 'y_fc_logits'
        with tf.variable_scope(name_y):
            logits = tf.layers.dense(y,
                                     1,
                                     kernel_initializer=init_kernel)
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
    tf.summary.histogram('{}/kernel_{}'.format(scope.split('/')[-1], layer),
                         net_vars[0], [name_model])
    tf.summary.histogram('{}/bias_{}'.format(scope.split('/')[-1], layer),
                         net_vars[1], [name_model])
    tf.summary.histogram('{}/act_{}'.format(scope.split('/')[-1], layer),
                         net, [name_model])
    layer += 1

def leakyReLu(x, alpha=0.1, name='leaky_relu'):
    """ Leaky relu """
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
