import os
from easydict import EasyDict as edict
import tensorflow as tf
import ops

#Define Encoder
encoder = edict()
encoder.channel = 64
encoder.n_enconder = 3
encoder.n_resblock = 4
encoder.weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
encoder.norm_fn = ops.instance_norm
encoder.dropout_ratio = 0.0

#Define Decoder
decoder = edict()
decoder.channel = 256
decoder.n_decoder = 3
decoder.n_resblock = 4
decoder.weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
decoder.norm_fn = ops.instance_norm
decoder.dropout_ratio = 0.0

#Define Discriminator
discriminator = edict()
discriminator.n_discriminator = 6
discriminator.channel = 64
discriminator.weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
discriminator.n_scales = 3

#Define Weights of Loss Function
loss = edict()
loss.gan_w = 1                      # weight of adversarial loss
loss.recon_x_w = 10                 # weight of image reconstruction loss
loss.recon_h_w = 0                  # weight of hidden reconstruction loss
loss.recon_kl_w = 0.01              # weight of KL loss for reconstruction
loss.recon_x_cyc_w = 10             # weight of cycle consistency loss
loss.recon_kl_cyc_w = 0.01          # weight of KL loss for cycle consistency
loss.vgg_w = 0.0                     # weight of domain-invariant perceptual loss
loss.perceptual_loss = edict({
    "network": "vgg_19",
    "weights": [1.0/32.0, 1.0/16.0, 1.0/8.0],
    "means": [123.68, 116.779, 103.939]
    })

#Define Attetion Network
attention = edict()
attention.add_attention = False

#Define path
path = edict()
path.data_root_path = "./datasets/horse2zebra"
path.outputs = "outputs"
path.pretrained_perceptual_loss_path = "./pretrained/vgg_19.ckpt"

#Define Learning Hyper-parameters
learning = edict()
learning.learning_rate = 1E-4
learning.weight_decay = 2E-5
learning.checkpoint_steps = 1000
learning.num_iters = 1000000
learning.beta1 = 0.5
learning.beta2 = 0.999
learning.batch_size = 2
learning.image_size = 256
learning.image_mean = 127.5
learning.image_std = 127.5
learning.summary = 250

