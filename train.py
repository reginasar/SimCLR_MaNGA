import tensorflow as tf
import datetime
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
import yaml
import time
import shutil
import numpy as np
#from astropy.io import fits
from cnn_big import BigCNN as TheCNN
from losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
from helpers import get_negative_mask
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from augmentations import random_rotate, gaussian_filter, duplicate,\
                                 randomize_error_pair, crop_reshape_center_pair
from helpers import rebin_nd
from parallel_help import load_pkl, pair_parallel_pipes, detect_missing_cubes

start0 = time.time()

input_shape = (32, 32, 5) #define input shape
epochs = 500
batch_size = 1024
temperature = 0.5


parallel_pipe_normed = load_pkl('parallel_pipe_normed.pkl')
manga_normed = load_pkl('manga_normed_sigma_clip.pkl')
existing_files = detect_missing_cubes(parallel_pipe_normed)
original, transformed, random_select = pair_parallel_pipes(manga_normed, parallel_pipe_normed, existing_files)

original_bin = np.zeros((original.shape[0], input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
transformed_bin = np.zeros_like(original_bin)
for ii in range(original.shape[0]):
    original_bin[ii, :, :, :] = rebin_nd(original[ii], [32, 32, 1])
    transformed_bin[ii, :, :, :] = rebin_nd(transformed[ii], [32, 32, 1])

train_dataset = tf.data.Dataset.from_tensor_slices((original_bin, transformed_bin))
print(train_dataset)
train_dataset = train_dataset.map(random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)
print(train_dataset)
train_dataset = train_dataset.map(randomize_error_pair, num_parallel_calls=tf.data.experimental.AUTOTUNE)
print(train_dataset)
train_dataset = train_dataset.map(crop_reshape_center_pair, num_parallel_calls=tf.data.experimental.AUTOTUNE)
print(train_dataset)
train_dataset = train_dataset.map(gaussian_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
print(train_dataset)

#prepare input data for training
train_dataset = train_dataset.repeat(epochs)
print(train_dataset)
train_dataset = train_dataset.shuffle(4096)
print(train_dataset)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
print(train_dataset)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

#model definition
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(3e-4)

model = TheCNN(10)

#model(tf.ones((1,32,32,3)))
#model_id = '20200429-161754'
#model_path = os.path.join('logs', model_id, 'train', 'checkpoints')
#model.load_weights(os.path.join(model_path, 'model.h5'))
#model = TheCNN(input_shape=input_shape, out_dim=10)

# Mask to remove positive examples from the batch of negative samples
negative_mask = get_negative_mask(batch_size)

#create dir to save model
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join('logs', current_time, 'train')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

#training step function
@tf.function
def train_step(xis, xjs):
    with tf.GradientTape() as tape:
        ris, zis = model(xis)
        rjs, zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        # tf.summary.histogram('zis', zis, step=optimizer.iterations)
        # tf.summary.histogram('zjs', zjs, step=optimizer.iterations)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (batch_size, 1))
        l_pos /= temperature
        # assert l_pos.shape == (config['batch_size'], 1), "l_pos shape not valid" + str(l_pos.shape)  # [N,1]

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(batch_size, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (batch_size, -1))
            l_neg /= temperature

            # assert l_neg.shape == (
            #     config['batch_size'], 2 * (config['batch_size'] - 1)), "Shape of negatives not expected." + str(
            #     l_neg.shape)
            logits = tf.concat([l_pos, l_neg], axis=1)  # [N,K+1]
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * batch_size)
        tf.summary.scalar('loss', loss, step=optimizer.iterations)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#    return loss

#train the model
step=0
with train_summary_writer.as_default():
    for xis, xjs in train_dataset:
        start = time.time()
        train_step(xis, xjs)

        step += 1
        if (step+1)%100 == 0:
            print(step+1,'step completed, total time:', (time.time()-start0)/60., 'min')
#        end = time.time()
#        print("Total time per batch:", end - start)

#save model
model_checkpoints_folder = os.path.join(train_log_dir, 'checkpoints')
if not os.path.exists(model_checkpoints_folder):
    os.makedirs(model_checkpoints_folder)

model.save_weights(os.path.join(model_checkpoints_folder, 'model.h5'))

print("-------------------------------")
print("TOTAL TIME:", (time.time()-start0)/60., "min")
print("-------------------------------")
print('trained on:', original_bin.shape[0], 'samples')
