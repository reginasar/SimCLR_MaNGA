import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0], new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def rebin_nd(arr, new_shape):
    new_arr = np.zeros((new_shape[0], new_shape[1], arr.shape[2]))
    for ii in range(arr.shape[2]):
        new_arr[:,:,ii] = rebin(arr[:,:,ii], new_shape)
    return new_arr

def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)

def match_cluster_labels(labels1, labels2):
    index_all = np.arange(labels1.shape[0])
    n_clus = np.max([labels1.max(), labels2.max()])+1
    cost = np.zeros((n_clus, n_clus))
    
    for hh in range(n_clus):
        for kk in range(n_clus):
            cost[hh, kk] = np.intersect1d(index_all[labels1==hh], index_all[labels2==kk], assume_unique=True).shape[0]
    
    new_labels2 = np.zeros_like(labels2)
    row_ind, col_ind = linear_sum_assignment(-cost[:, :])
    #   print(row_ind, col_ind)
    for hh in range(n_clus):
        new_labels2[labels2==col_ind[hh]] = hh
    return labels1, new_labels2


