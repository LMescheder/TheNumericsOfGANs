import tensorflow as tf
from consensus_gan.models import (
    conv3, conv4,
    dcgan3, dcgan4,
    dcgan3_nobn, dcgan4_nobn,
    dcgan3_nobn_cf, dcgan4_nobn_cf,
    resnet,
)

generator_dict = {
    'conv3': conv3.generator,
    'conv4': conv4.generator,
    'dcgan3': dcgan3.generator,
    'dcgan4': dcgan4.generator,
    'dcgan3_nobn': dcgan3_nobn.generator,
    'dcgan4_nobn': dcgan4_nobn.generator,
    'dcgan3_nobn_cf': dcgan3_nobn_cf.generator,
    'dcgan4_nobn_cf': dcgan4_nobn_cf.generator,
    'resnet': resnet.generator,
}

discriminator_dict = {
    'conv3': conv3.discriminator,
    'conv4': conv4.discriminator,
    'dcgan3': dcgan3.discriminator,
    'dcgan4': dcgan4.discriminator,
    'dcgan3_nobn': dcgan3_nobn.discriminator,
    'dcgan4_nobn': dcgan4_nobn.discriminator,
    'dcgan3_nobn_cf': dcgan3_nobn_cf.discriminator,
    'dcgan4_nobn_cf': dcgan4_nobn_cf.discriminator,
    'resnet': resnet.discriminator,
}


def get_generator(model_name, scope='generator', **kwargs):
    model_func = generator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)


def get_discriminator(model_name, scope='discriminator', **kwargs):
    model_func = discriminator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)
