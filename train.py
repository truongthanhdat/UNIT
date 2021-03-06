import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import UNIT
import params
from utils import Timer, load_images, collect_summaries, load_pretrained_perceptual_loss
import scipy.misc


if __name__ == "__main__":

    image_a_train_path = os.path.join(params.path.data_root_path, "trainA")
    image_b_train_path = os.path.join(params.path.data_root_path, "trainB")
    image_a_path_list = [os.path.join(image_a_train_path, image_name) for image_name in os.listdir(image_a_train_path)]
    image_b_path_list = [os.path.join(image_b_train_path, image_name) for image_name in os.listdir(image_b_train_path)]

    batch_shape = [params.learning.batch_size, params.learning.image_size, params.learning.image_size, 3]
    image_a = tf.placeholder(shape = batch_shape, dtype = tf.float32)
    image_b = tf.placeholder(shape = batch_shape, dtype = tf.float32)

    results = UNIT.unit(image_a, image_b, add_attention = params.attention.add_attention, is_training = True)

    gen_loss = results.gen_loss + results.L1_loss + results.KL_loss +  results.perceptual_loss
    adv_loss = results.adv_loss

    summaries = collect_summaries(results)
    summaries.append(tf.summary.scalar("total_gen_loss", gen_loss))
    summaries.append(tf.summary.scalar("total_adv_loss", adv_loss))

    encoder_vars = [var for var in slim.get_variables() if "encoder" in var.name]
    decoder_vars = [var for var in slim.get_variables() if "decoder" in var.name]
    discriminator_vars = [var for var in slim.get_variables() if "discriminator" in var.name]
    generator_vars = encoder_vars + decoder_vars

    print("Number of Generator Variabels: {}".format(len(generator_vars)))
    print("Number of Discriminator Variabels: {}".format(len(discriminator_vars)))
    print("Number of Training Image:\n\tDomain A: {}\n\tDomain B: {}".format(len(image_a_path_list), len(image_b_path_list)))



    optimizer_G = tf.train.AdamOptimizer(params.learning.learning_rate, params.learning.beta1, params.learning.beta2)
    optimizer_D = tf.train.AdamOptimizer(params.learning.learning_rate, params.learning.beta1, params.learning.beta2)
    G_step_op = optimizer_G.minimize(gen_loss, var_list = generator_vars)
    D_step_op = optimizer_D.minimize(adv_loss, var_list = discriminator_vars)

    summaries_op = tf.summary.merge(summaries)
    summary_writer = tf.summary.FileWriter(params.path.outputs, graph=tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(params.path.outputs)
    if latest_checkpoint is not None:
        saver.restore(sess, latest_checkpoint)

    if params.loss.vgg_w > 0.0:
        load_pretrained_perceptual_loss(sess = sess, path = params.path.pretrained_perceptual_loss_path)

    timer = Timer()
    counter = 0
    for iter in range(params.learning.num_iters):
        timer.tic()
        _image_a, _image_b = load_images(params.learning.batch_size, image_a_path_list, image_b_path_list)

        feed_dict = {
                image_a: _image_a,
                image_b: _image_b
                }

        #Update Discriminator
        _, summary = sess.run([D_step_op, summaries_op], feed_dict = feed_dict)

        if (iter + 1) % params.learning.summary == 0:
            summary_writer.add_summary(summary, counter)
            counter += 1

        #Update Generator
        _, summary = sess.run([G_step_op, summaries_op], feed_dict = feed_dict)
        if (iter + 1) % params.learning.summary == 0:
            summary_writer.add_summary(summary, counter)
            counter += 1

        print("[{:07d}/{:07d}] Elapsed time in update: {}".format(iter + 1, params.learning.num_iters, timer.toc()))

        if (iter + 1) % params.learning.checkpoint_steps == 0:
            saver.save(sess, os.path.join(params.path.outputs, "model.ckpt"))
            print("Saving checkpoint at iteration {:07d}".format(iter + 1))

