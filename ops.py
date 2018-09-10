import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import utils
import params
import vgg

def leaky_relu(inputs):
    return tf.nn.leaky_relu(inputs, 0.01)

def relu(inputs):
    return tf.nn.relu(inputs)

def tanh(inputs):
    return tf.nn.tanh(inputs)

def sigmoid(inputs):
    return tf.nn.sigmoid(inputs)

def batch_norm(inputs, is_training = True, scope = "batch_norm"):
     return tf_contrib.layers.batch_norm(x, decay = 0.9, epsilon = 1e-05,
                                        center = True, scale = True,
                                        is_training = is_training, scope = scope)

def instance_norm(x, scope='instance', is_training = True) :
    return tf.contrib.layers.instance_norm(x, epsilon = 1e-05,
                                           center = True, scale = True,
                                           scope = scope)


def conv(inputs, dim, kernel_size, stride = 2,
            activation_fn = leaky_relu, is_training = True,
            weights_initializer = initializers.xavier_initializer(), scope = "conv_0"):
    """
        Define Convolutional Layer
    """

    with tf.variable_scope(scope):
        net = slim.conv2d(inputs, dim, kernel_size, stride,
                            activation_fn = activation_fn, #is_training = is_training,
                            weights_initializer = weights_initializer)
        return net

def deconv(inputs, dim, kernel_size, stride,
            activation_fn = leaky_relu, is_training = True,
            weights_initializer = initializers.xavier_initializer(), scope = "deconv_0"):
    """
        Define Deconvolutional Layer
    """

    with tf.variable_scope(scope):
        net = slim.conv2d_transpose(inputs, dim, kernel_size, stride,
                                    activation_fn = activation_fn, #is_training = is_training,
                                    weights_initializer = weights_initializer)
        return net

def resblock(inputs, dim, kernel_size = [3, 3], stride = 1, dropout_ratio = 0.0,
                is_training = True, norm_fn = instance_norm, scope='resblock_0',
                weights_initializer = initializers.xavier_initializer()):
    """
        Define Residual Block
    """
    with tf.variable_scope(scope):
        with tf.variable_scope("res_1"):
            net = slim.conv2d(inputs, dim, kernel_size, stride,
                                activation_fn = relu, #is_training = is_training,
                                normalizer_fn = norm_fn, normalizer_params = {"is_training": is_training},
                                weights_initializer = weights_initializer)

        with tf.variable_scope("res_2"):
            net = slim.conv2d(net, dim, kernel_size, stride,
                                activation_fn = relu, #is_training = is_training,
                                normalizer_fn = None, normalizer_params = {"is_training": is_training},
                                weights_initializer = weights_initializer)

        #with tf.variable_scope("res_3"):
        #    net = slim.conv2d(net, dim, kernel_size, stride,
        #                        activation_fn = relu, #is_training = is_training,
        #                        normalizer_fn = norm_fn, normalizer_params = {"is_training": is_training},
        #                        weights_initializer = weights_initializer)

        net = slim.dropout(net, keep_prob = 1 - dropout_ratio, is_training = is_training)

        return net + inputs

def gaussian_noise_layer(mu):
    sigma = 1.0
    gaussian_random_vector = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
    return mu + sigma * gaussian_random_vector


def KL_divergence(mu) :
    # KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, axis = -1)
    # loss = tf.reduce_mean(KL_divergence)
    mu_2 = tf.square(mu)
    loss = tf.reduce_mean(mu_2)
    return loss

def L1_loss(x, y) :
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss

def multiply_mask(images, mask):
    r, g, b = tf.split(images, 3, 3)
    r = tf.multiply(r, mask)
    g = tf.multiply(g, mask)
    b = tf.multiply(b, mask)
    return tf.concat([r, g, b], axis = 3)



def perceptual_loss(real, fake, network = "vgg_16"):
    if params.loss.vgg_w <= 0.0:
        return 0.0

    real = real * params.learning.image_std + params.learning.image_mean
    fake = fake * params.learning.image_std + params.learning.image_mean
    real = utils.perceptual_loss_image_preprocess(real)
    fake = utils.perceptual_loss_image_preprocess(fake)
    image = tf.concat([real, fake], axis = 0)

    with tf.variable_scope("perceptual_loss"):
        if network == "vgg_16":
            with slim.arg_scope(vgg.vgg_arg_scope()):
                conv1, conv2, conv3 = vgg.vgg_16(image)
        elif network == "vgg_19":
            with slim.arg_scope(vgg.vgg_arg_scope()):
                conv1, conv2, conv3 = vgg.vgg_19(image)
        else:
            raise NotImplementedError("")

        losses = []
        for i, features in enumerate([conv1, conv2, conv3]):
            real, fake = tf.split(features, 2, 0)
            losses.append( params.loss.perceptual_loss.weights[i] * tf.reduce_mean( tf.square(real - fake) ) )

        return losses[0] + losses[1] + losses[2]


def discriminator_loss(real, fake, smoothing=False, use_lsgan=True) :
    if use_lsgan :
        if smoothing :
            real_loss = tf.reduce_mean(tf.squared_difference(real, 0.9)) * 0.5
        else :
            real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0)) * 0.5

        fake_loss = tf.reduce_mean(tf.square(fake)) * 0.5
    else :
        if smoothing :
            real_labels = tf.fill(tf.shape(real), 0.9)
        else :
            real_labels = tf.ones_like(real)

        fake_labels = tf.zeros_like(fake)

        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

    loss = real_loss + fake_loss

    return loss

def generator_loss(fake, smoothing=False, use_lsgan=True) :
    if use_lsgan :
        if smoothing :
            loss = tf.reduce_mean(tf.squared_difference(fake, 0.9)) * 0.5
        else :
            loss = tf.reduce_mean(tf.squared_difference(fake, 1.0)) * 0.5
    else :
        if smoothing :
            fake_labels = tf.fill(tf.shape(fake), 0.9)
        else :
            fake_labels = tf.ones_like(fake)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

    return loss

