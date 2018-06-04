import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import gan.mnist_utilities as network
import data.mnist as data
from utils.evaluations import do_prc
from sklearn.metrics import precision_recall_fscore_support


RANDOM_SEED = 146
FREQ_PRINT = 20 # print frequency image tensorboard [20]
STEPS_NUMBER = 500

def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter

def display_parameters(batch_size, starting_lr, ema_decay,
                       weight, method, degree, label):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Method for discriminator: ', method)
    print('Degree for L norms: ', degree)
    print('Anomalous label: ', label)

def display_progression_epoch(j, id_max):
    '''See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def create_logdir(method, weight, label, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "bigan/train_logs/mnist/{}/{}/{}/{}".format(weight, method, label, rd)


def train_and_test(nb_epochs, weight, method, degree, random_seed, label):
    """ Runs the Bigan on the KDD dataset

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        method (str, optional): 'fm' for ``Feature Matching`` or "cross-e"
                                     for ``cross entropy``, "efm" etc.
        anomalous_label (int): int in range 0 to 10, is the class/digit
                                which is considered outlier
    """
    logger = logging.getLogger("GAN.train.mnist.{}.{}".format(method,label))

    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    trainx, trainy = data.get_train(label, True)
    trainx_copy = trainx.copy()
    testx, testy = data.get_test(label, True)

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.999

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    logger.info('Building training graph...')

    logger.warn("The GAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree, label)

    gen = network.generator
    dis = network.discriminator

    # Sample noise from random normal distribution
    random_z = tf.random_normal([batch_size, latent_dim], mean=0.0, stddev=1.0, name='random_z')
    # Generate images with generator
    generator = gen(random_z, is_training=is_training_pl)
    # Pass real and fake images into discriminator separately
    real_d, inter_layer_real = dis(input_pl, is_training=is_training_pl)
    fake_d, inter_layer_fake = dis(generator, is_training=is_training_pl, reuse=True)

    with tf.name_scope('loss_functions'):
        # Calculate seperate losses for discriminator with real and fake images
        real_discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(1, shape=[batch_size]), real_d, scope='real_discriminator_loss')
        fake_discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(0, shape=[batch_size]), fake_d, scope='fake_discriminator_loss')
        # Add discriminator losses
        discriminator_loss = real_discriminator_loss + fake_discriminator_loss
        # Calculate loss for generator by flipping label on discriminator output
        generator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(1, shape=[batch_size]), fake_d, scope='generator_loss')

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        update_ops_dis = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')

        with tf.control_dependencies(update_ops_gen): # attached op for moving average batch norm
            gen_op = optimizer_gen.minimize(generator_loss, var_list=gvars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(discriminator_loss, var_list=dvars)

        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

    with tf.name_scope('training_summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('real_discriminator_loss', real_discriminator_loss, ['dis'])
            tf.summary.scalar('fake_discriminator_loss', fake_discriminator_loss, ['dis'])
            tf.summary.scalar('discriminator_loss', discriminator_loss, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', generator_loss, ['gen'])

        with tf.name_scope('image_summary'):
            tf.summary.image('reconstruct', generator, 8, ['image'])
            tf.summary.image('input_images', input_pl, 8, ['image'])


        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_im = tf.summary.merge_all('image')

    logger.info('Building testing graph...')

    with tf.variable_scope("latent_variable"):
        z_optim = tf.get_variable(name='z_optim', shape= [batch_size, latent_dim], initializer=tf.truncated_normal_initializer())
        reinit_z = z_optim.initializer
    # EMA
    generator_ema = gen(z_optim, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)
    # Pass real and fake images into discriminator separately
    real_d_ema, inter_layer_real_ema = dis(input_pl, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)
    fake_d_ema, inter_layer_fake_ema = dis(generator_ema, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)

    with tf.name_scope('error_loss'):
        delta = input_pl - generator_ema
        delta_flat = tf.contrib.layers.flatten(delta)
        gen_score = tf.norm(delta_flat, ord=degree, axis=1, keep_dims=False, name='epsilon')

    with tf.variable_scope('Discriminator_loss'):
        if method == "cross-e":
            dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_d_ema), logits=fake_d_ema)

        elif method == "fm":
            fm = inter_layer_real_ema - inter_layer_fake_ema
            fm = tf.contrib.layers.flatten(fm)
            dis_score = tf.norm(fm, ord=degree, axis=1, keep_dims=False,
                             name='d_loss')

        dis_score = tf.squeeze(dis_score)

    with tf.variable_scope('Total_loss'):
        loss = (1 - weight) * gen_score + weight * dis_score

    with tf.variable_scope("Test_learning_rate"):
        step = tf.Variable(0, trainable=False)
        boundaries = [200, 300]
        values = [0.01, 0.001, 0.0005]
        learning_rate_invert = tf.train.piecewise_constant(step, boundaries, values)
        reinit_lr = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope="Test_learning_rate"))

    with tf.name_scope('Test_optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate_invert).minimize(loss, global_step=step, var_list=[z_optim], name='optimizer')
        reinit_optim = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope='Test_optimizer'))

    reinit_test_graph_op = [reinit_z, reinit_lr, reinit_optim]

    with tf.name_scope("Scores"):
        list_scores = loss

    logdir = create_logdir(method, weight, label, random_seed)

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
                             save_model_secs=120)

    logger.info('Start training...')
    with sv.managed_session() as sess:

        logger.info('Initialization done')

        writer = tf.summary.FileWriter(logdir, sess.graph)

        train_batch = 0
        epoch = 0

        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr

            begin = time.time()
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling unl dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]

            train_loss_dis, train_loss_gen = [0, 0]
            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)

                # construct randomly permuted minibatches
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {input_pl: trainx[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, ld, sm = sess.run([train_dis_op, discriminator_loss, sum_op_dis], feed_dict=feed_dict)
                train_loss_dis += ld
                writer.add_summary(sm, train_batch)

                # train generator
                feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, lg, sm = sess.run([train_gen_op, generator_loss, sum_op_gen], feed_dict=feed_dict)
                train_loss_gen += lg
                writer.add_summary(sm, train_batch)

                if t % FREQ_PRINT == 0:  # inspect reconstruction
                    t= np.random.randint(0,4000)
                    ran_from = t
                    ran_to = t + batch_size
                    sm = sess.run(sum_op_im, feed_dict={input_pl: trainx[ran_from:ran_to],is_training_pl: False})
                    writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_dis /= nr_batches_train

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_dis))

            epoch += 1

        logger.warn('Testing evaluation...')
        inds = rng.permutation(testx.shape[0])
        testx = testx[inds]  # shuffling unl dataset
        testy = testy[inds]
        scores = []
        inference_time = []

        # testing
        for t in range(nr_batches_test):
            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            # invert the gan
            feed_dict = {input_pl: testx[ran_from:ran_to],
                         is_training_pl:False}

            for step in range(STEPS_NUMBER):
                _ = sess.run(optimizer, feed_dict=feed_dict)
            scores += sess.run(list_scores, feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_val_batch)
            sess.run(reinit_test_graph_op)

        logger.info('Testing : mean inference time is %.4f' % (
            np.mean(inference_time)))
        ran_from = nr_batches_test * batch_size
        ran_to = (nr_batches_test + 1) * batch_size
        size = testx[ran_from:ran_to].shape[0]
        fill = np.ones([batch_size - size, 28, 28, 1])

        batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
        feed_dict = {input_pl: batch,
                     is_training_pl: False}

        for step in range(STEPS_NUMBER):
            _ = sess.run(optimizer, feed_dict=feed_dict)
        batch_score = sess.run(list_scores,
                           feed_dict=feed_dict).tolist()

        scores += batch_score[:size]

        prc_auc = do_prc(scores, testy,
               file_name=r'gan/mnist/{}/{}/{}'.format(method, weight,
                                                     label),
               directory=r'results/gan/mnist/{}/{}/'.format(method,
                                                           weight))

        print("Testing | PRC AUC = {:.4f}".format(prc_auc))

def run(nb_epochs, weight, method, degree, label, random_seed=42):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, degree, random_seed, label)
