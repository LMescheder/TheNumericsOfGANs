import numpy as np
import tensorflow as tf
from tqdm import tqdm
from consensus_gan.utils import *
from consensus_gan import ops
from consensus_gan.optimizer import (
    ConsensusOptimizer, SimGDOptimizer, AltGDOptimizer, ClipOptimizer, SmoothingOptimizer
)
from consensus_gan.inception_score import InceptionScore
import ipdb
import time

def train(generator, discriminator, x_real, config):
    batch_size = config['batch_size']
    output_size = config['output_size']
    c_dim = config['c_dim']
    z_dim = config['z_dim']

    # TODO: fix that this has to be run before all other graph building ops
    if config['is_inception_scores']:
        inception_scorer = InceptionScore(config['inception_dir'])

    x_real = 2.*x_real - 1.
    z = tf.random_normal([batch_size, z_dim])
    x_fake = generator(z)
    x_fake_test = generator(z, is_training=False)
    d_out_real = discriminator(x_real)
    d_out_fake = discriminator(x_fake)

    # GAN / Divergence type
    g_loss, d_loss =  get_losses(d_out_real, d_out_fake, x_real, x_fake, discriminator, config)

    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    # Optimizer
    optimizer = get_optimizer(config, global_step)

    g_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    d_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    train_ops = optimizer.conciliate(d_loss, g_loss, d_vars, g_vars, global_step=global_step)

    time_diff = tf.placeholder(tf.float32)
    Wall_clock_time = tf.Variable(0., trainable=False)
    update_Wall_op = Wall_clock_time.assign_add(time_diff)

    # Summaries
    summaries = [
        tf.summary.scalar('loss/discriminator', d_loss),
        tf.summary.scalar('loss/generator', g_loss),
        tf.summary.scalar('loss/Wall_clock_time', Wall_clock_time),
    ]
    summary_op = tf.summary.merge(summaries)

    # Inception scores
    inception_scores = tf.placeholder(tf.float32)
    inception_mean, inception_var = tf.nn.moments(inception_scores, [0])
    inception_summary_op = tf.summary.merge([
         tf.summary.scalar('inception_score/mean', inception_mean),
         tf.summary.scalar('inception_score/std', tf.sqrt(inception_var)),
         tf.summary.scalar('inception_score/Wall_clock_time', Wall_clock_time),
         tf.summary.histogram('inception_score/histogram', inception_scores)
    ])

    # Supervisor
    sv = tf.train.Supervisor(
        logdir=config['log_dir'], global_step=global_step,
        summary_op=summary_op, save_summaries_secs=15,
    )

    z_test_np = np.random.randn(batch_size, z_dim)

    with sv.managed_session() as sess:
        # Show real data
        samples = sess.run(x_real)
        samples = 0.5*(samples+1.)
        save_images(samples[:64], [8, 8], config['sample_dir'], 'real.png')

        progress = tqdm(range(config['nsteps']))

        for batch_idx in progress:
            if sv.should_stop():
               break

            niter = sess.run(global_step)

            t0 = time.time()

            # Train
            for train_op in train_ops:
                sess.run(train_op)

            t1 = time.time()
            sess.run(update_Wall_op, feed_dict={time_diff: t1 - t0})

            d_loss_out, g_loss_out = sess.run([d_loss, g_loss])

            progress.set_description('Loss_g: %4.4f, Loss_d: %4.4f'
                % (g_loss_out, d_loss_out))
            sess.run(global_step_op)

            if np.mod(niter, config['ntest']) == 0:
                # Test
                samples = sess.run(x_fake_test, feed_dict={z: z_test_np})
                samples = 0.5*(samples+1.)

                save_images(samples[:64], [8, 8], os.path.join(config['sample_dir'], 'samples'),
                            'train_{:06d}.png'.format(niter)
                )
                # Inception scores
                if config['is_inception_scores']:
                    inception_scores_np = get_inception_score(sess, inception_scorer, x_fake_test)
                    inception_scores_summary_out = sess.run(inception_summary_op, feed_dict={inception_scores: inception_scores_np})
                    sv.summary_computed(sess, inception_scores_summary_out)

def get_losses(d_out_real, d_out_fake, x_real, x_fake, discriminator, config):
    batch_size = config['batch_size']
    gan_type = config['gan_type']

    if gan_type == 'standard':
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.ones_like(d_out_fake)
        ))
    elif gan_type == 'JS':
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake
    elif gan_type == 'KL':
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(-d_out_fake)
    elif gan_type == 'tv':
        d_loss = tf.reduce_mean(tf.tanh(d_out_fake) - tf.tanh(d_out_real))
        g_loss = tf.reduce_mean(-tf.tanh(d_out_fake))
    elif gan_type == 'indicator':
        d_loss = tf.reduce_mean(d_out_fake - d_out_real)
        g_loss = tf.reduce_mean(-d_out_fake)
    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % gan_type)

    return g_loss, d_loss



def get_optimizer(config, global_step):
    optimizer_name = config['optimizer']
    reg_param = config['reg_param']
    learning_rate = config['learning_rate']
    nsteps = config['nsteps']

    learning_rate_decayed = learning_rate #tf.train.exponential_decay(learning_rate, global_step, nsteps, 0.01)

    if optimizer_name == 'simgd':
        optimizer = SimGDOptimizer(learning_rate_decayed)
    elif optimizer_name == 'altgd':
        optimizer = AltGDOptimizer(learning_rate_decayed, g_steps=config['altgd_gsteps'], d_steps=config['altgd_dsteps'])
    elif optimizer_name == 'conopt':
        optimizer = ConsensusOptimizer(learning_rate_decayed, alpha=reg_param)
    elif optimizer_name == 'clip':
        optimizer = ClipOptimizer(learning_rate_decayed, alpha=reg_param)
    elif optimizer_name == 'smooth':
        optimizer = SmoothingOptimizer(learning_rate_decayed, alpha=reg_param)

    return optimizer


def get_inception_score(sess, inception_scorer, x_fake):
    all_samples = []
    for i in range(100):
        all_samples.append(sess.run(x_fake))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    return inception_scorer.get_inception_score(sess, list(all_samples))
