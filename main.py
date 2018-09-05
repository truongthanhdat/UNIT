import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import UNIT
import params
import utils
import scipy.misc

def collect_summaries(results):
    summaries = []
    for item in results.items():
        if "image" in item[0]:
            summaries.append(tf.summary.image(item[0], item[1]))
        elif "loss" in item[0]:
            summaries.append(tf.summary.scalar(item[0], item[1]))

    return summaries

def load_images(batch_size, a_list, b_list):
    a_list = np.random.choice(a_list, batch_size, False)
    b_list = np.random.choice(b_list, batch_size, False)
    images_a = []
    images_b = []
    for i in range(batch_size):
        image_a = scipy.misc.imread(a_list[i], mode = "RGB")
        image_b = scipy.misc.imread(b_list[i], mode = "RGB")
        image_a = scipy.misc.imresize(image_a, (params.image_size, params.image_size))
        image_b = scipy.misc.imresize(image_b, (params.image_size, params.image_size))
        image_a = image_a.astype(np.float32) / 255
        image_b = image_b.astype(np.float32) / 255
        images_a.append(image_a)
        images_b.append(image_b)

    images_a = np.concatenate(images_a, 0)
    images_b = np.concatenate(images_b, 0)

    if len(images_a.shape) == 3:
        images_a = np.expand_dims(images_a, axis = 0)
    if len(images_b.shape) == 3:
        images_b = np.expand_dims(images_b, axis = 0)

    return images_a, images_b


if __name__ == "__main__":

    image_a_train_path = os.path.join(params.data_root_path, "trainA")
    image_b_train_path = os.path.join(params.data_root_path, "trainB")
    image_a_path_list = [os.path.join(image_a_train_path, image_name) for image_name in os.listdir(image_a_train_path)]
    image_b_path_list = [os.path.join(image_b_train_path, image_name) for image_name in os.listdir(image_b_train_path)]

    image_a = tf.placeholder(shape = [params.batch_size, params.image_size, params.image_size, 3], dtype = tf.float32)
    image_b = tf.placeholder(shape = [params.batch_size, params.image_size, params.image_size, 3], dtype = tf.float32)
    results = UNIT.unit(image_a, image_b)

    gen_loss = results.gen_loss + results.L1_loss + results.KL_loss
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



    optimizer_G = tf.train.AdamOptimizer(params.learning_rate, params.beta1, params.beta2)
    optimizer_D = tf.train.AdamOptimizer(params.learning_rate, params.beta1, params.beta2)
    G_step_op = optimizer_G.minimize(gen_loss, var_list = generator_vars)
    D_step_op = optimizer_D.minimize(adv_loss, var_list = discriminator_vars)

    summaries_op = tf.summary.merge(summaries)
    summary_writer = tf.summary.FileWriter(params.outputs, graph=tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(params.outputs)
    if latest_checkpoint is not None:
        saver.restore(sess, latest_checkpoint)

    timer = utils.Timer()
    counter = 0
    for iter in range(params.num_iters):
        timer.tic()
        _image_a, _image_b = load_images(params.batch_size, image_a_path_list, image_b_path_list)

        feed_dict = {
                image_a: _image_a,
                image_b: _image_b
                }
        #Update Discriminator
        _, summary = sess.run([D_step_op, summaries_op], feed_dict = feed_dict)
        summary_writer.add_summary(summary, counter)
        counter += 1
        #Update Generator
        _, summary = sess.run([G_step_op, summaries_op], feed_dict = feed_dict)
        summary_writer.add_summary(summary, counter)
        counter += 1

        print("[{:06d}/{:06d}]\tElapsed time in update: {}".format(iter + 1, params.num_iters, timer.toc()))

        if (iter + 1) % params.checkpoint_steps == 0:
            saver.save(sess, os.path.join(params.outputs, "model.ckpt"))
            print("Saving checkpoint at iteration {:06d}".format(iter + 1))

