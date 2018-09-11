import time
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim

def get_variables(scope):
    return [var for var in slim.get_variables() if scope in var.name]

def normalize_image(images):
    images = (images + 1.0) / 2.0 * 255.0
    images = tf.cast(images, tf.uint8)
    return images

def collect_summaries(results):
    summaries = []
    for item in results.items():
        if "loss" in item[0]:
            summaries.append(tf.summary.scalar(item[0], item[1]))

    image_a = tf.concat([results.image_a, results.image_a2a, results.image_a2b2a, results.image_a2b], axis = 2)
    image_b = tf.concat([results.image_b, results.image_b2b, results.image_b2a2b, results.image_b2a], axis = 2)
    summaries.append(tf.summary.image("image_a", normalize_image(image_a)))
    summaries.append(tf.summary.image("image_b", normalize_image(image_b)))

    if results.has_key("mask_a2b"):
        mask_a = tf.concat([results.mask_a2b, results.mask_a2b2a], axis = 2)
        mask_b = tf.concat([results.mask_b2a, results.mask_b2a2b], axis = 2)
        summaries.append(tf.summary.image("mask_a", tf.cast(mask_a * 255, tf.uint8)))
        summaries.append(tf.summary.image("mask_b", tf.cast(mask_b * 255, tf.uint8)))

    return summaries

def load_images(batch_size, image_size, a_list, b_list):
    a_list = np.random.choice(a_list, batch_size, False)
    b_list = np.random.choice(b_list, batch_size, False)
    images_a = []
    images_b = []
    for i in range(batch_size):
        image_a = scipy.misc.imread(a_list[i], mode = "RGB")
        image_b = scipy.misc.imread(b_list[i], mode = "RGB")
        image_a = scipy.misc.imresize(image_a, (image_size, image_size))
        image_b = scipy.misc.imresize(image_b, (image_size, image_size))
        image_a = image_a.astype(np.float32)/255.0 * 2.0 - 1.0
        image_b = image_b.astype(np.float32)/255.0 * 2.0 - 1.0
        images_a.append(np.expand_dims(image_a, 0))
        images_b.append(np.expand_dims(image_b, 0))

    images_a = np.concatenate(images_a, 0)
    images_b = np.concatenate(images_b, 0)

    if len(images_a.shape) == 3:
        images_a = np.expand_dims(images_a, axis = 0)
    if len(images_b.shape) == 3:
        images_b = np.expand_dims(images_b, axis = 0)

    return images_a, images_b


class Timer:
    def __init__(self):
        self._tic = time.time()

    def tic(self):
        self._tic = time.time()

    def toc(self):
        return time.time() - self._tic


