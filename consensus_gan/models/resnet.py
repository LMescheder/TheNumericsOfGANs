import tensorflow as tf
from tensorflow.contrib import slim
import logging
from consensus_gan.ops import lrelu

logger = logging.getLogger(__name__)


def generator(z, f_dim, output_size, c_dim, is_training=True):
    # Network
    net = slim.fully_connected(z, output_size//16 * output_size//16 *  8 * f_dim, activation_fn=None)
    net = tf.reshape(net, [-1, output_size//16, output_size//16, 8 * f_dim])

    argscope_conv2d_trp = slim.arg_scope(
        [slim.conv2d_transpose], kernel_size=[5, 5], stride=[2, 2], activation_fn=None)
    argscope_conv2d = slim.arg_scope(
        [slim.conv2d], kernel_size=[3, 3], stride=[1, 1], activation_fn=None)

    with argscope_conv2d, argscope_conv2d_trp:
        net = slim.conv2d_transpose(lrelu(net), 4 * f_dim)
        dnet = slim.conv2d(lrelu(net), 4 * f_dim//2)
        net += 1e-1 * slim.conv2d(lrelu(dnet), 4 * f_dim)

        net = slim.conv2d_transpose(lrelu(net), 2 * f_dim)
        dnet = slim.conv2d(lrelu(net), 2*f_dim//2)
        net += 1e-1 * slim.conv2d(lrelu(dnet), 2*f_dim)
        
        net = slim.conv2d_transpose(lrelu(net),  f_dim)
        dnet = slim.conv2d(lrelu(net), f_dim//2)
        net += 1e-1 * slim.conv2d(lrelu(dnet), f_dim)

        net = slim.conv2d_transpose(lrelu(net), c_dim)

    out = tf.nn.tanh(net)

    return out


def discriminator(x, f_dim, output_size, c_dim, is_training=True):
    # Network
    net = x

    argscope_conv2d = slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=[2, 2],
                                     activation_fn=None)
    with argscope_conv2d:
        net = slim.conv2d(net, f_dim)
        dnet = slim.conv2d(lrelu(net), f_dim//2, kernel_size=[3,3], stride=[1,1])
        net += 1e-1 * slim.conv2d(lrelu(dnet), f_dim, kernel_size=[3,3], stride=[1,1])
        
        net = slim.conv2d(lrelu(net), 2 * f_dim)
        dnet = slim.conv2d(lrelu(net), 2 * f_dim//2, kernel_size=[3,3], stride=[1,1])
        net += 1e-1 * slim.conv2d(lrelu(dnet),  2* f_dim, kernel_size=[3,3], stride=[1,1])
        
        net = slim.conv2d(lrelu(net), 4 * f_dim)
        dnet = slim.conv2d(lrelu(net), 4 * f_dim//2, kernel_size=[3,3], stride=[1,1])
        net += 1e-1 * slim.conv2d(lrelu(dnet), 4 * f_dim, kernel_size=[3,3], stride=[1,1])
                                  
        net = slim.conv2d(lrelu(net), 8 * f_dim)
        dnet = slim.conv2d(lrelu(net), 8 * f_dim//2, kernel_size=[3,3], stride=[1,1])
        net += 1e-1 * slim.conv2d(lrelu(dnet), 8 * f_dim, kernel_size=[3,3], stride=[1,1])
        
    net = tf.reshape(net, [-1, output_size//16 * output_size//16 * 8 * f_dim])
    logits = slim.fully_connected(lrelu(net), 1, activation_fn=None)
    logits = tf.squeeze(logits, -1)

    return logits
