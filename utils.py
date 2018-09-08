import time
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import params

def normalize_image(images):
    images = images * params.learning.image_std + params.learning.image_mean
    images = tf.cast(images, tf.uint8)
    return images

def collect_summaries(results):
    summaries = []
    for item in results.items():
        if "image" in item[0]:
            summaries.append(tf.summary.image(item[0], normalize_image(item[1])))
        elif "loss" in item[0]:
            summaries.append(tf.summary.scalar(item[0], item[1]))
        elif "mask" in item[0]:
            summaries.append(tf.summary.scalar(item[0], tf.cast(item[1] * 255, tf.uint8)))

    return summaries

def load_images(batch_size, a_list, b_list):
    a_list = np.random.choice(a_list, batch_size, False)
    b_list = np.random.choice(b_list, batch_size, False)
    images_a = []
    images_b = []
    for i in range(batch_size):
        image_a = scipy.misc.imread(a_list[i], mode = "RGB")
        image_b = scipy.misc.imread(b_list[i], mode = "RGB")
        image_a = scipy.misc.imresize(image_a, (params.learning.image_size, params.learning.image_size))
        image_b = scipy.misc.imresize(image_b, (params.learning.image_size, params.learning.image_size))
        image_a = (image_a.astype(np.float32) - params.learning.image_mean) / params.learning.image_std
        image_b = (image_b.astype(np.float32) - params.learning.image_mean) / params.learning.image_std
        images_a.append(np.expand_dims(image_a, 0))
        images_b.append(np.expand_dims(image_b, 0))

    images_a = np.concatenate(images_a, 0)
    images_b = np.concatenate(images_b, 0)

    if len(images_a.shape) == 3:
        images_a = np.expand_dims(images_a, axis = 0)
    if len(images_b.shape) == 3:
        images_b = np.expand_dims(images_b, axis = 0)

    return images_a, images_b

def load_pretrained_perceptual_loss(sess, path, remove_first_scope = True):
    variables = slim.get_variables(scope = "perceptual_loss")
    restored_vars = {}
    for var in variables:
        name = var.name.split(":")[0]
        if remove_first_scope:
            name = "/".join(name.split("/")[1:])
        restored_vars[name] = var

    saver = tf.train.Saver(restored_vars)
    saver.restore(sess, path)

def perceptual_loss_image_preprocess(image):
    channels = tf.split(axis = 3, num_or_size_splits = 3, value = image)
    for i in range(3):
        channels[i] -= params.loss.perceptual_loss.means[i]
    return tf.concat(axis = 3, values = channels)

class Timer:
    def __init__(self):
        self._tic = time.time()

    def tic(self):
        self._tic = time.time()

    def toc(self):
        return time.time() - self._tic


