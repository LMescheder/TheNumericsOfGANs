import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import logging
from collections import defaultdict
from consensus_gan.ops import lrelu

logger = logging.getLogger(__name__)

def generator(z, f_dim, output_size, c_dim, is_training=True):
    bn_kwargs = {
        'is_training': is_training, 'updates_collections': None
    }

    # Network
    net = slim.fully_connected(z, output_size//8 * output_size//8 * 4*f_dim,
        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=bn_kwargs
        )
    net = tf.reshape(net, [-1, output_size//8, output_size//8, 4*f_dim])

    conv2d_trp_argscope =  slim.arg_scope([slim.conv2d_transpose],
        kernel_size=[5,5], stride=[2,2], activation_fn=tf.nn.relu, normalizer_params=bn_kwargs,
    )
    with conv2d_trp_argscope:
        net = slim.conv2d_transpose(net, 2*f_dim, normalizer_fn=slim.batch_norm)
        net = slim.conv2d_transpose(net, f_dim, normalizer_fn=slim.batch_norm)
        net = slim.conv2d_transpose(net, c_dim, activation_fn=None)

    out = tf.nn.tanh(net)

    return out

def discriminator(x, f_dim, output_size, c_dim, is_training=True):
    bn_kwargs = {
        'is_training': is_training, 'updates_collections': None
    }

    # Network
    net = x

    conv2d_argscope =  slim.arg_scope([slim.conv2d],
        kernel_size=[5,5], stride=[2,2], activation_fn=lrelu, normalizer_params=bn_kwargs
    )
    with conv2d_argscope:
        net = slim.conv2d(net, f_dim)
        net = slim.conv2d(net, 2*f_dim, normalizer_fn=slim.batch_norm)
        net = slim.conv2d(net, 4*f_dim, normalizer_fn=slim.batch_norm)

    net = tf.reshape(net, [-1, output_size//8 * output_size//8 * 4 * f_dim])
    logits = slim.fully_connected(net, 1, activation_fn=None, normalizer_fn=None)
    logits = tf.squeeze(logits, -1)

    return logits
