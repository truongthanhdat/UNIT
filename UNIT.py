import network
import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
from easydict import EasyDict
import utils

def unit_encoder(image, scope, params, is_training = True):
    return network.encoder(image, scope = scope, is_training = is_training,
                    shared_scope = "shared_encoder", channels = params.channels,
                    n_conv = params.n_conv, n_resblock = params.n_resblock)

def unit_decoder(z, scope, params, is_training):
    return network.decoder(z, scope = scope, is_training = is_training,
                    shared_scope = "shared_decoder", channels = params.channels,
                    n_conv = params.n_conv, n_resblock = params.n_resblock)

def unit_multiscale_discriminator(image, scope, params, is_training):
    return network.muliscale_discriminator(image, scope = scope, is_training = is_training,
                    shared_scope = "shared_discriminator", channels = params.channels,
                    n_conv = params.n_conv, gan_type = params.gan_type, n_scale = params.n_scale)

def unit(image_a, image_b, params, is_training = True):
    #Encode Image
    z_a = unit_encoder(image_a, "encoder_A", params.encoder, is_training)
    z_b = unit_encoder(image_b, "encoder_B", params.encoder, is_training)
    #Decode Image
    image_a2a = unit_decoder(z_a, "decoder_A", params.decoder, is_training)
    image_b2b = unit_decoder(z_b, "decoder_B", params.decoder, is_training)
    #Decode Image Cross Domain
    image_a2b = unit_decoder(z_a, "decoder_B", params.decoder, is_training)
    image_b2a = unit_decoder(z_a, "decoder_A", params.decoder, is_training)
    #Encode Image Again
    z_a2b = unit_encoder(image_a2b, "encoder_B", params.encoder, is_training)
    z_b2a = unit_encoder(image_b2a, "encoder_A", params.encoder, is_training)
    #Decode Image Again
    image_a2b2a = unit_decoder(z_a2b, "decoder_A", params.decoder, is_training)
    image_b2a2b = unit_decoder(z_b2a, "decoder_B", params.decoder, is_training)
    #Discriminator
    real_a = unit_multiscale_discriminator(image_a, "discriminator_A", params.discriminator, is_training)
    fake_a = unit_multiscale_discriminator(image_a2b, "discriminator_A", params.discriminator, is_training)
    real_b = unit_multiscale_discriminator(image_b, "discriminator_B", params.discriminator, is_training)
    fake_b = unit_multiscale_discriminator(image_b2a, "discriminator_B", params.discriminator, is_training)

    #LOSS
    L1_loss = ops.L1_loss(image_a, image_a2a) + ops.L1_loss(image_b, image_b2b)
    L1_loss_CC = ops.L1_loss(image_a, image_a2b2a) + ops.L1_loss(image_b, image_b2a2b)
    KL_loss = ops.KL_loss(z_a) + ops.KL_loss(z_b)
    KL_loss_CC = ops.KL_loss(z_a2b) + ops.KL_loss(z_b2a)
    gen_loss = ops.generator_loss(fake_a + fake_b, params.discriminator.gan_type)
    adv_loss = ops.discriminator_loss(real_a + real_b, fake_a + fake_b, params.discriminator.gan_type)

    return EasyDict({
        "image_a": image_a, "image_b": image_b,
        "image_a2a": image_a2a, "image_b2b": image_b2b,
        "image_a2b": image_a2b, "image_b2a": image_b2a,
        "image_a2b2a": image_a2b2a, "image_b2a2b": image_b2a2b,
        "L1_loss": L1_loss, "L1_loss_CC": L1_loss_CC,
        "KL_loss": KL_loss, "KL_loss_CC": KL_loss_CC,
        "gen_loss": gen_loss, "adv_loss": adv_loss
        })

if __name__ == "__main__":
    params = EasyDict()
    params.encoder = EasyDict({
        "n_conv": 3,
        "n_resblock": 4,
        "channels": 64
        })
    params.decoder = EasyDict({
        "n_conv": 3,
        "n_resblock": 4,
        "channels": 64
        })
    params.discriminator = EasyDict({
        "n_conv": 4,
        "channels": 64,
        "gan_type": "lsgan",
        "n_scale": 3
        })
    image_a = tf.placeholder(tf.float32, [1, 256, 256, 3])
    image_b = tf.placeholder(tf.float32, [1, 256, 256, 3])
    results = unit(image_a, image_b, params, True)
    for item in results.items():
        print(item)

    var_E = utils.get_variables("encoder")
    var_G = utils.get_variables("decoder")
    var_D = utils.get_variables("discriminator")

    print("Number of encoder {}".format(len(var_E)))
    print("Number of decoder {}".format(len(var_G)))
    print("Number of discriminator {}".format(len(var_D)))

    for var in var_E + var_G + var_D:
        print var



