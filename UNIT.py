import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import utils
import params
import ops
from easydict import EasyDict as edict

def encoder(inputs, scope = "encoder", is_training = True, reuse = False):
    with tf.variable_scope(scope, reuse = reuse):
        channel = params.encoder.channel
        net = inputs
        net = ops.conv(net, scope = "conv1",
                dim = channel, kernel_size = [7, 7], stride = 1,
                activation_fn = ops.leaky_relu, is_training = is_training,
                weights_initializer = params.encoder.weights_initializer)
        for i in range(1, params.encoder.n_enconder):
            channel *= 2
            net = ops.conv(net, scope = "conv_{}".format(i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 2,
                    activation_fn = ops.leaky_relu, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer)
        for i in range(params.encoder.n_resblock):
            net = ops.resblock(net, scope = "resblock_{}".format(i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 1,
                    norm_fn = params.encoder.norm_fn, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer,
                    dropout_ratio = params.encoder.dropout_ratio)

        net = ops.gaussian_noise_layer(net)
        return net

def decoder(inputs, scope = "decoder", is_training = True, reuse = False):
    with tf.variable_scope(scope, reuse = reuse):
        channel = params.decoder.channel
        net = inputs
        for i in range(params.decoder.n_resblock):
            net = ops.resblock(net, scope = "resblock_{}".format(params.decoder.n_resblock - i),
                    dim = channel, kernel_size = [3, 3], stride = 1,
                    norm_fn = params.encoder.norm_fn, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer,
                    dropout_ratio = params.encoder.dropout_ratio)

        for i in range(1, params.decoder.n_decoder):
            channel = channel / 2
            net = ops.deconv(net, scope = "deconv_{}".format(params.decoder.n_decoder - i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 2,
                    activation_fn = ops.leaky_relu, is_training = is_training,
                    weights_initializer = params.decoder.weights_initializer)
        net = ops.deconv(net, scope = "deconv_1",
                    dim = 3, kernel_size = [1, 1], stride = 1,
                    activation_fn = ops.tanh, is_training = is_training,
                    weights_initializer = params.decoder.weights_initializer)
        return net

def discriminator(inputs, scope = "discriminator", is_training = True, reuse = False):
    with tf.variable_scope(scope, reuse = reuse):
        net = inputs
        channel = params.discriminator.channel
        for i in range(params.discriminator.n_discriminator - 1):
            net = ops.conv(net, scope = "conv_{}".format(i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 2,
                    activation_fn = ops.leaky_relu, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer)
            channel *= 2

        net = ops.conv(net, scope = "conv_6",
                    dim = 1, kernel_size = [1, 1], stride = 1,
                    activation_fn = None, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer)
        #Final Conv Layer uses Sigmoid
        return net

def unit(image_a, image_b):
    results = edict()
    #Encode Image
    results.z_a = encoder(image_a, scope = "encoder_A", reuse = False)
    results.z_b = encoder(image_b, scope = "encoder_B", reuse = False)
    z = tf.concat([results.z_a, results.z_b], axis = 0)
    #Decode Image
    results.image_a2a = decoder(results.z_a, "decoder_A", reuse = False)
    results.image_b2b = decoder(results.z_b, "decoder_B", reuse = False)
    #Decode Cross Domain
    results.image_a2b = decoder(results.z_a, "decoder_B", reuse = True)
    results.image_b2a = decoder(results.z_b, "decoder_A", reuse = True)
    #Encode Again
    results.z_ba = encoder(results.image_b2a, scope = "encoder_A", reuse = True)
    results.z_ab = encoder(results.image_a2b, scope = "encoder_B", reuse = True)
    #Decode Again
    results.image_a2b2a = decoder(results.z_ab, scope = "decoder_A", reuse = True)
    results.image_b2a2b = decoder(results.z_ba, scope = "decoder_B", reuse = True)
    #Discriminator
    results.real_a = discriminator(image_a, "discriminator_A", reuse = False)
    results.fake_a = discriminator(results.image_b2a, "discriminator_A", reuse = True)
    results.real_b = discriminator(image_b, "discriminator_B", reuse = False)
    results.fake_b = discriminator(results.image_a2b, "discriminator_B", reuse = True)

    #Build Loss
    results.L1_loss = params.loss.recon_x_w * (ops.L1_loss(image_a, results.image_a2a) + ops.L1_loss(image_b, results.image_a2b)) + \
              params.loss.recon_x_cyc_w * (ops.L1_loss(image_a, results.image_a2b2a) + ops.L1_loss(image_b, results.image_b2a2b))

    results.KL_loss = params.loss.recon_kl_w * (ops.KL_divergence(results.z_a) + ops.KL_divergence(results.z_b)) + \
              params.loss.recon_kl_cyc_w * (ops.KL_divergence(results.z_ab) + ops.KL_divergence(results.z_ba))

    results.gen_loss = params.loss.gan_w * (ops.generator_loss(results.fake_a) + ops.generator_loss(results.fake_b))

    results.adv_loss = params.loss.gan_w * (ops.discriminator_loss(results.real_a, results.fake_a) + ops.discriminator_loss(results.real_b, results.fake_b))

    results.image_a = image_a
    results.image_b = image_b

    return results

if __name__ == "__main__":
    image_a = tf.placeholder(shape = [1, 256, 256, 3], dtype = tf.float32)
    image_b = tf.placeholder(shape = [1, 256, 256, 3], dtype = tf.float32)

    results = unit(image_a, image_b)
    for item in results.items():
        print(item)

