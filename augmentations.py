import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from gaussian_filter import GaussianBlur
from helpers import rebin_nd

#----functions for single tensor-----

def duplicate(v):
    v2 = tf.identity(v)
    return v, v2

def cutout(img):
    cut_side1 = 2 * tf.random.uniform(shape=(), maxval=5, dtype=tf.int32) + 2
    cut_side2 = 2 * tf.random.uniform(shape=(), maxval=5, dtype=tf.int32) + 2
    margin1 = int(img.shape[0]/2.) - cut_side1
    margin2 = int(img.shape[1]/2.) - cut_side2
    margin10 = int(img.shape[0]/2.) + cut_side1
    margin20 = int(img.shape[1]/2.) + cut_side2
    img_reduced = img[margin1:margin10, margin2:margin20, :]
    img_reduced = tf.expand_dims(img_reduced, 0)
    img_reduced = tfa.image.random_cutout(img_reduced, (cut_side1,cut_side2))
    img_reduced = tf.reshape(img_reduced, (2*cut_side1, 2*cut_side2, img_reduced.shape[3]))
#    paddings = tf.constant([[margin1, margin1], [margin2, margin2], [0,0]])
    ones_window = tf.pad(tf.ones_like(img_reduced), [[margin1, margin1], [margin2, margin2], [0,0]], "CONSTANT")
    img = img - img * ones_window + tf.pad(img_reduced, [[margin1, margin1], [margin2, margin2], [0,0]], "CONSTANT")
    return img

def crop_reshape_centered(img):
    percent = tf.random.uniform(shape=(), minval=0.25, maxval=0.75)
    shape = tf.shape(img)
    if percent > 0.5:
        new_side = int((percent+0.25) * img.shape[0])
        margin10 = int((img.shape[0]-new_side)/2)
        margin20 = new_side + margin10
        img_crop = img[margin10:margin20, margin10:margin20, :]
    #    img_crop = tf.reshape(img_crop, (1, new_side, new_side, shape[2]))
        img_new = tf.image.resize(img_crop, [img.shape[0], img.shape[0]], \
            method='bicubic', preserve_aspect_ratio=True,\
            antialias=False)
    else:
        margin = int(percent * img.shape[0])
        #margin = int((percent-0.25) * img.shape[0])
        img_big = tf.pad(img, [[margin, margin], [margin, margin], [0,0]], "CONSTANT")
        img_new = tf.image.resize(img_big, [img.shape[0], img.shape[0]], \
            method='bilinear', preserve_aspect_ratio=True,\
            antialias=True)    

    img_new = tf.reshape(img_new, (shape[0], shape[1], shape[2]))    

    return img_new

def crop_reshape(img):
    percent = tf.random.uniform(shape=(), minval=0.5, maxval=1)
    shape = tf.shape(img)
    new_side = int(percent *img.shape[0])

    limit = int(img.shape[0]-new_side)
    margin10 = tf.random.uniform(shape=[], minval=0, maxval=limit, dtype=tf.int32)
    margin20 = new_side + margin10
    margin11 = tf.random.uniform(shape=[], minval=0, maxval=limit, dtype=tf.int32)
    margin21 = new_side + margin11
    img_crop = img[margin10:margin20, margin11:margin21, :]

#    img_crop = tf.reshape(img_crop, (1, new_side, new_side, shape[2]))
    img_new = tf.image.resize(img_crop, [img.shape[0], img.shape[0]], \
        method='bicubic', preserve_aspect_ratio=True,\
        antialias=False)

    img_new = tf.reshape(img_new, (shape[0], shape[1], shape[2]))

    return img_new

def randomize_with_error(img):
    noise_factor = np.load('SNR_weights.npy', 'r')
    Vimg_delta = np.expand_dims(np.ones((noise_factor.shape[0], noise_factor.shape[1])), axis=2) * 0.1
    noise_factor = np.concatenate((Vimg_delta, noise_factor), axis=2)
    noise_factor = rebin_nd(noise_factor, [img.shape[0], img.shape[1], 1])
    random_factor = tf.random.normal(shape=img.shape, mean=tf.ones(img.shape), stddev=noise_factor, dtype=tf.float32)
    random_factor = tf.clip_by_value(random_factor, 0, 10)
    img_new = random_factor * img
    img_new = tf.clip_by_value(img_new, 0, 1)

    return img_new

def randomize_with_error_3(img):
    noise_factor = np.load('SNR_weights.npy', 'r')
    noise_factor = noise_factor[:,:,2:]
    Vimg_delta = np.expand_dims(np.ones((noise_factor.shape[0], noise_factor.shape[1])), axis=2) * 0.1
    noise_factor = np.concatenate((Vimg_delta, noise_factor), axis=2)
    noise_factor = rebin_nd(noise_factor, [img.shape[0], img.shape[1], 1])
    random_factor = tf.random.normal(shape=img.shape, mean=tf.ones(img.shape), stddev=noise_factor, dtype=tf.float32)
    random_factor = tf.clip_by_value(random_factor, 0, 10)
    img_new = random_factor * img
    img_new = tf.clip_by_value(img_new, 0, 1)

    return img_new

def randomize_with_error_4(img):
    noise_factor = np.load('SNR_weights.npy', 'r')
    noise_factor = rebin_nd(noise_factor, [img.shape[0], img.shape[1], 1])
    random_factor = tf.random.normal(shape=img.shape, mean=tf.ones(img.shape), stddev=noise_factor, dtype=tf.float32)
    random_factor = tf.clip_by_value(random_factor, 0, 10)
    img_new = random_factor * img
    img_new = tf.clip_by_value(img_new, 0, 1)

    return img_new

def random_mask(img):
    shape = img.shape
    a = tf.random.uniform(shape=img[:,:,1:].shape, minval=0, maxval=2, dtype=tf.int32)
    b = tf.random.uniform(shape=a.shape, minval=0, maxval=2, dtype=tf.int32)
    c = tf.random.uniform(shape=a.shape, minval=0, maxval=2, dtype=tf.int32)
    mask = -((a * b * c) - 1)
    mask = tf.cast(mask, dtype=tf.float32)
    img_new = img[:,:,1:] * mask
    img_new = tf.concat((tf.expand_dims(img[:,:,0], 2), img_new), axis=2)
    return img_new

#----funtions for paired tensors-----

def gaussian_filter(v1, v2):
    shape = tf.shape(v2)
    side = v2.shape[1] 
    k_size = np.int((side * 0.5) * 2 + 1) # kernel size is set to be 10% of the image height/width and odd
    gaussian_ope = GaussianBlur(kernel_size=k_size, min=0.1, max=2.0)
#    v1 = tf.py_function(gaussian_ope, [v1], [tf.float32])
    v2 = tf.py_function(gaussian_ope, [v2], [tf.float32])
#    v1 = tf.reshape(v1, (side, side, shape[2]))
    v2 = tf.reshape(v2, (side, side, shape[2]))
    return v1, v2

def random_rotate(v1, v2):
    v1 = tf.image.rot90(v1, tf.random.uniform(shape=[], maxval=4, dtype=tf.int32))
    v2 = tf.image.rot90(v2, tf.random.uniform(shape=[], maxval=4, dtype=tf.int32))

    v1 = tf.image.random_flip_left_right(v1, seed=None)
    v2 = tf.image.random_flip_left_right(v2, seed=None)

    v1 = tf.image.random_flip_up_down(v1, seed=None)
    v2 = tf.image.random_flip_up_down(v2, seed=None)

    return v1, v2

def crop_reshape_pair(v1, v2):
#    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
#    if prob < 0.5:
#        v1 = crop_reshape(v1)

    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
    if prob < 0.5:
        v2 = crop_reshape(v2)
    return v1, v2

def crop_reshape_center_pair(v1, v2):
#    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
#    if prob < 0.5:
#        v1 = crop_reshape_centered(v1)

    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
    if prob < 0.5:
        v2 = crop_reshape_centered(v2)
    return v1, v2

def cutout_pair(v1, v2):
#    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
#    if prob < 0.5:
#        v1 = cutout(v1)

    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
    if prob < 0.5:
        v2 = cutout(v2)
    return v1, v2

def random_mask_pair(v1, v2):
#    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
#    if prob < 0.5:
#        v1 = random_mask(v1)

    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
    if prob < 0.5:
        v2 = random_mask(v2)
    return v1, v2

def randomize_error_pair(v1, v2):
#    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
#    if prob < 0.5:
#        v1 = randomize_with_error(v1)

    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
    if prob < 0.5:
        v2 = randomize_with_error(v2)
    return v1, v2

def randomize_error_pair_3(v1, v2):
#    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
#    if prob < 0.5:
#        v1 = randomize_with_error(v1)

    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
    if prob < 0.5:
        v2 = randomize_with_error_3(v2)
    return v1, v2

def randomize_error_pair_4(v1, v2):
#    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
#    if prob < 0.5:
#        v1 = randomize_with_error(v1)

    prob = tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)
    if prob < 0.5:
        v2 = randomize_with_error_4(v2)
    return v1, v2
