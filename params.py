import os
from easydict import EasyDict as edict
import tensorflow as tf
import ops

encoder = edict()
decoder = edict()
discriminator = edict()
attention = edict()
loss = edict()

encoder.channel = 64
encoder.n_enconder = 3
encoder.n_resblock = 4
encoder.weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
encoder.norm_fn = ops.instance_norm
encoder.dropout_ratio = 0.0

decoder.channel = 256
decoder.n_decoder = 3
decoder.n_resblock = 4
decoder.weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
decoder.norm_fn = ops.instance_norm
decoder.dropout_ratio = 0.0

discriminator.n_discriminator = 6
discriminator.channel = 64

loss.gan_w = 1                      # weight of adversarial loss
loss.recon_x_w = 10                 # weight of image reconstruction loss
loss.recon_h_w = 0                  # weight of hidden reconstruction loss
loss.recon_kl_w = 0.01              # weight of KL loss for reconstruction
loss.recon_x_cyc_w = 10             # weight of cycle consistency loss
loss.recon_kl_cyc_w = 0.01          # weight of KL loss for cycle consistency
loss.vgg_w = 1                      # weight of domain-invariant perceptual loss

pretrained_vgg_path = "./pretrained/vgg_16.ckpt"
means = [123.68, 116.779, 103.939]

data_root_path = "./datasets/horse2zebra"
outputs = "outputs"
learning_rate = 1E-4
weight_decay = 2E-5
checkpoint_steps = 1000
num_iters = 1000000
beta1 = 0.5
beta2 = 0.999
batch_size = 2
image_size = 256

image_mean = 127.5
image_std = 127.5
