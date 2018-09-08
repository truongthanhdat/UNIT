import time
import numpy as np
import scipy.misc
import tensorflow as tf
import params

def normalize_image(images):
    images = images * params.image_std + params.image_mean
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
        image_a = scipy.misc.imresize(image_a, (params.image_size, params.image_size))
        image_b = scipy.misc.imresize(image_b, (params.image_size, params.image_size))
        image_a = (image_a.astype(np.float32) - params.image_mean) / params.image_std
        image_b = (image_b.astype(np.float32) - params.image_mean) / params.image_std
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


