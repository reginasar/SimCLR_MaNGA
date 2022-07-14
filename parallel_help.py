import numpy as np
import _pickle as cPickle


def load_pkl(filename):
    with open(filename, "rb") as input_file:
        data = cPickle.load(input_file)
    return data


def pair_parallel_pipes(data1, data2, existing_files):
    random_select = np.random.choice(2, data1.shape[0], p=[0.75, 0.25])
    random_select *= existing_files
    #print(existing_files[:50])
    copy = np.zeros_like(data1)
    for ii in range(data1.shape[0]):
        if random_select[ii]==1:
            copy[ii,:,:,:] = data2[ii,:,:,:]
        else:
            copy[ii,:,:,:] = data1[ii,:,:,:]
    return data1, copy, random_select

def detect_missing_cubes(data):
    existing_files = np.zeros(data.shape[0], dtype=np.int8)
    for ii in range(data.shape[0]):
        existing_files[ii] = np.any(np.where(data[ii,:,:,:]!=0., 1, 0))
    return existing_files

