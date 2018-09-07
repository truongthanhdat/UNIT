import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import utils
import params
import ops
from easydict import EasyDict as edict
from network import encoder, decoder, attention, discriminator

def unit(image_a, image_b, is_training = True, add_attention = True):
    #Encode Image
    z_a = encoder(image_a, scope = "encoder_A",
                    reuse = False, is_training = is_training,
                    shared_scope = "shared_encoder", shared_reuse = False)
    z_b = encoder(image_b, scope = "encoder_B",
                    reuse = False, is_training = is_training,
                    shared_scope = "shared_encoder", shared_reuse = True)

    if add_attention:
        #Compute Mask
        mask_a2b = attention(image_a, scope = "attention_A2B",
                        reuse = False, is_training = is_training,
                        shared_scope = "shared_attention", shared_reuse = False)
        mask_b2a = attention(image_b, scope = "attention_B2A",
                        reuse = False, is_training = is_training,
                        shared_scope = "shared_attention", shared_reuse = True)

    #Decode Image
    image_a2a = decoder(z_a, "decoder_A",
                    reuse = False, is_training = is_training,
                    shared_scope = "shared_decoder", shared_reuse = False)
    image_b2b = decoder(z_b, "decoder_B",
                    reuse = False, is_training = is_training,
                    shared_scope = "shared_decoder", shared_reuse = True)

    #Decode Cross Domain
    image_a2b = decoder(z_a, "decoder_B",
                reuse = True, is_training = is_training,
                shared_scope = "shared_decoder", shared_reuse = True)
    image_b2a = decoder(z_b, "decoder_A",
                reuse = True, is_training = is_training,
                shared_scope = "shared_decoder", shared_reuse = True)
    if add_attention:
        image_a2b = ops.multiply_mask(image_a, mask_a2b) + ops.multiply_mask(image_a2b, 1 - mask_a2b)
        image_b2a = ops.multiply_mask(image_b, mask_b2a) + ops.multiply_mask(image_b2a, 1 - mask_b2a)

    #Encode Again
    z_b2a = encoder(image_b2a, scope = "encoder_A",
                reuse = True, is_training = is_training,
                shared_scope = "shared_encoder", shared_reuse = True)
    z_a2b = encoder(image_a2b, scope = "encoder_B",
                reuse = True, is_training = is_training,
                shared_scope = "shared_encoder", shared_reuse = True)

    if add_attention:
        #Compute Translated Mask
        mask_a2b2a = attention(image_a2b, scope = "attention_B2A",
                            reuse = True, is_training = is_training,
                            shared_scope = "shared_attention", shared_reuse = True)
        mask_b2a2b = attention(image_b2a, scope = "attention_A2B",
                            reuse = True, is_training = is_training,
                            shared_scope = "shared_attention", shared_reuse = True)

    #Decode Again
    image_a2b2a = decoder(z_a2b, scope = "decoder_A",
                reuse = True, is_training = is_training,
                shared_scope = "shared_decoder", shared_reuse = True)
    image_b2a2b = decoder(z_b2a, scope = "decoder_B",
                reuse = True, is_training = is_training,
                shared_scope = "shared_decoder", shared_reuse = True)
    if add_attention:
        image_a2b2a = ops.multiply_mask(image_a2b, mask_a2b2a) + ops.multiply_mask(image_a2b2a, 1 - mask_a2b2a)
        image_b2a2b = ops.multiply_mask(image_b2a, mask_b2a2b) + ops.multiply_mask(image_b2a2b, 1 - mask_b2a2b)


    #Discriminator
    real_a = discriminator(image_a, "discriminator_A",
                reuse = False, is_training = is_training,
                shared_scope = "shared_discriminator", shared_reuse = False)
    fake_a = discriminator(image_b2a, "discriminator_A",
                reuse = True, is_training = is_training,
                shared_scope = "shared_discriminator", shared_reuse = True)

    real_b = discriminator(image_b, "discriminator_B",
                reuse = False, is_training = is_training,
                shared_scope = "shared_discriminator", shared_reuse = True)
    fake_b = discriminator(image_a2b, "discriminator_B",
                 reuse = True, is_training = is_training,
                shared_scope = "shared_discriminator", shared_reuse = True)


    #Build Loss
    L1_loss = params.loss.recon_x_w * (ops.L1_loss(image_a, image_a2a) + ops.L1_loss(image_b, image_a2b)) + \
              params.loss.recon_x_cyc_w * (ops.L1_loss(image_a, image_a2b2a) + ops.L1_loss(image_b, image_b2a2b))

    KL_loss = params.loss.recon_kl_w * (ops.KL_divergence(z_a) + ops.KL_divergence(z_b)) + \
              params.loss.recon_kl_cyc_w * (ops.KL_divergence(z_a2b) + ops.KL_divergence(z_b2a))

    gen_loss = params.loss.gan_w * (ops.generator_loss(fake_a) + ops.generator_loss(fake_b))

    adv_loss = params.loss.gan_w * (ops.discriminator_loss(real_a, fake_a) + ops.discriminator_loss(real_b, fake_b))

    perceptual_loss = ops.perceptual_loss(
                tf.concat([image_a, image_b], axis = 0),
                tf.concat([image_a2b, image_b2a], axis = 0)
            )

    if add_attention:
        return edict({
            "image_a": image_a, "image_b": image_b,
            "image_a2a": image_a2a, "image_a2b": image_b2b,
            "image_a2b": image_a2b, "image_b2a": image_b2a,
            "image_a2b2a": image_a2b2a, "image_b2a2b": image_b2a2b,
            "mask_a2b": mask_a2b, "mask_b2a": mask_b2a,
            "mask_a2b2a": mask_a2b2a, "mask_b2a2a": mask_b2a2b,
            "L1_loss": L1_loss, "KL_loss": KL_loss, "perceptual_loss": perceptual_loss,
            "gen_loss": gen_loss, "adv_loss": adv_loss
        })
    else:
        return edict({
            "image_a": image_a, "image_b": image_b,
            "image_a2a": image_a2a, "image_a2b": image_b2b,
            "image_a2b": image_a2b, "image_b2a": image_b2a,
            "image_a2b2a": image_a2b2a, "image_b2a2b": image_b2a2b,
            "L1_loss": L1_loss, "KL_loss": KL_loss, "perceptual_loss": perceptual_loss,
            "gen_loss": gen_loss, "adv_loss": adv_loss
        })

if __name__ == "__main__":
    image_a = tf.placeholder(shape = [1, 256, 256, 3], dtype = tf.float32)
    image_b = tf.placeholder(shape = [1, 256, 256, 3], dtype = tf.float32)

    results = unit(image_a, image_b, add_attention = False)
    for item in results.items():
        print(item)

    variables = slim.get_variables()
    for var in variables:
        print(var.name)
    print("Number of Variables: {}".format(len(variables)))

