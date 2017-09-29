import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.training import moving_averages


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def get_batch_moments(x, is_training=True, decay=0.99, is_init=False):
    x_shape = x.get_shape()
    axis = list(range(len(x_shape)-1))

    moving_mean = tf.get_variable('moving_mean', x_shape[-1:], tf.float32, trainable=False,
                                  initializer=tf.zeros_initializer())
    moving_variance = tf.get_variable('moving_var', x_shape[-1:], tf.float32, trainable=False,
                                      initializer=tf.ones_initializer())

    if is_init:
        mean, variance = tf.nn.moments(x, axis)
    elif is_training:
        # Calculate the moments based on the individual batch.
        mean, variance = tf.nn.moments(x, axis, shift=moving_mean)
        # Update the moving_mean and moving_variance moments.
        update_moving_mean = moving_mean.assign_sub((1 - decay) * (moving_mean - mean))
        update_moving_variance = moving_variance.assign_sub(
            (1 - decay) * (moving_variance - variance))
        # Make sure the updates are computed here.
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            mean, variance = tf.identity(mean), tf.identity(variance)
    else:
        mean, variance = moving_mean, moving_variance
    return mean, tf.sqrt(variance + 1e-8)


def get_input_moments(x, is_init=False, name=None):
    '''Input normalization'''
    with tf.variable_scope(name, default_name='input_norm'):
        if is_init:
            # data based initialization of parameters
            mean, variance = tf.nn.moments(x, [0])
            std = tf.sqrt(variance + 1e-8)
            mean0 = tf.get_variable('mean0', dtype=tf.float32,
                                    initializer=mean, trainable=False)
            std0 = tf.get_variable('std0', dtype=tf.float32,
                                   initializer=std, trainable=False)
            return mean, std

        else:
            mean0 = tf.get_variable('mean0')
            std0 = tf.get_variable('std0')
            tf.assert_variables_initialized([mean0, std0])
            return mean0, std0


@add_arg_scope
def fully_connected(x, num_outputs, activation_fn=None,
                    init_scale=1., is_init=False, ema=None, name=None):
    ''' fully connected layer '''
    with tf.variable_scope(name, default_name='Full'):
        if is_init:
            # data based initialization of parameters
            V = tf.get_variable('V', [int(x.get_shape()[1]), num_outputs], tf.float32,
                                tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.matmul(x, V_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=-m_init * scale_init, trainable=True)
            x_init = scale_init * (x_init - m_init)
            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init

        else:
            V = tf.get_variable('V')
            g = tf.get_variable('g')
            b = tf.get_variable('b')

            tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            x = tf.matmul(x, V)
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            x = scaler * x + b

            # apply activation_fn
            if activation_fn is not None:
                x = activation_fn(x)
            return x


@add_arg_scope
def conv2d(x, num_outputs, kernel_size=[3, 3], stride=[1, 1], pad='SAME', activation_fn=None,
           init_scale=1., is_init=False, ema=None, name=None):
    ''' convolutional layer '''
    norm_axes = [0, 1, 2]
    with tf.variable_scope(name, default_name='Conv2D'):
        if is_init:
            # data based initialization of parameters
            V = tf.get_variable('V', kernel_size + [int(x.get_shape()[-1]), num_outputs],
                                tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            x_init = tf.nn.conv2d(x, V_norm, [1] + stride + [1], pad)
            m_init, v_init = tf.nn.moments(x_init, norm_axes)
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=-m_init * scale_init, trainable=True)
            x_init = scale_init * (x_init - m_init)
            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init

        else:
            v = tf.get_variable('V')
            g = tf.get_variable('g')
            b = tf.get_variable('b')

            # use weight normalization (salimans & kingma, 2016)
            w = tf.nn.l2_normalize(v, [0, 1, 2])

            # calculate convolutional layer output
            x = g * tf.nn.conv2d(x, w, [1] + stride + [1], pad)
            x = x + b

            # apply activation_fn
            if activation_fn is not None:
                x = activation_fn(x)
            return x


@add_arg_scope
def conv2d_transpose(x, num_outputs, kernel_size=[3, 3], stride=[1, 1],
                     pad='SAME', activation_fn=None,
                     init_scale=1., is_init=False, ema=None, name=None):
    ''' transposed convolutional layer '''
    xs = int_shape(x)
    norm_axes = [0, 1, 2]
    if pad == 'SAME':
        target_shape = [xs[0], xs[1] * stride[0],
                        xs[2] * stride[1], num_outputs]
    else:
        target_shape = [xs[0], xs[1] * stride[0] + kernel_size[0] -
                        1, xs[2] * stride[1] + kernel_size[1] - 1, num_outputs]
    with tf.variable_scope(name, default_name='Conv2DTrp'):
        if is_init:
            # data based initialization of parameters
            v = tf.get_variable('V', kernel_size + [num_outputs, int(x.get_shape()[-1])],
                                tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            v_norm = tf.nn.l2_normalize(v.initialized_value(), [0, 1, 3])
            x_init = tf.nn.conv2d_transpose(x, v_norm, target_shape, [1] + stride + [1],
                                            padding=pad)
            m_init, v_init = tf.nn.moments(x_init, norm_axes)
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=-m_init * scale_init, trainable=True)
            x_init = scale_init * (x_init - m_init)
            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init

        else:
            v = tf.get_variable('V')
            g = tf.get_variable('g')
            b = tf.get_variable('b')

            tf.assert_variables_initialized([v, g, b])

            # use weight normalization (salimans & kingma, 2016)
            w = tf.nn.l2_normalize(v, [0, 1, 3])

            # calculate convolutional layer output
            x = g * tf.nn.conv2d_transpose(x, w, target_shape, [1] + stride + [1], padding=pad)
            x = x + b

            # apply activation_fn
            if activation_fn is not None:
                x = activation_fn(x)
            return x


def upsample(x):
    xshape = [int(t) for t in x.get_shape()]
    # ipdb.set_trace()
    x_rs = tf.reshape(x, [xshape[0]*xshape[1], 1, xshape[2]*xshape[3]])
    x_rs = tf.tile(x_rs, [1, 2, 1])
    x_rs = tf.reshape(x_rs, [xshape[0]*2*xshape[1]*xshape[2], 1, xshape[3]])
    x_rs = tf.tile(x_rs, [1, 2, 1])
    x_out = tf.reshape(x_rs, [xshape[0], 2*xshape[1], 2*xshape[2], xshape[3]])

    return x_out


def get_var_maybe_avg(var_name, ema,  **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v


def get_vars_maybe_avg(var_names, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars


def int_shape(x):
    return list(map(int, x.get_shape()))
