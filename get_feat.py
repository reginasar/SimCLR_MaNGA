import numpy as np
from cnn_big import BigCNN as TheCNN
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf

def next_batch(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i: i+batch_size]# / 255.
        y_batch = y[i: i+batch_size]
        yield X_batch.astype(np.float32), y_batch

def get_features(model_id, X_maps):
    n_features = 1024
    batch_size = 1024
    X_simclr = np.zeros((X_maps.shape[0], n_features))
    y = np.arange(X_simclr.shape[0])

    #model_path = os.path.join('logs', model_id, 'train', 'checkpoints')
    model_path = 'model_' + model_id
    model_CNN = TheCNN(out_dim=1)
    model_CNN(tf.ones((1, X_maps.shape[1], X_maps.shape[2], X_maps.shape[3])))
    model_CNN.load_weights(os.path.join(model_path, 'model.h5'))
    
    X_simclr = []
    
    for batch_x, batch_y in next_batch(X_maps, y, batch_size):
        features_x, _ = model_CNN(batch_x)
        X_simclr.extend(features_x.numpy())
    
    X_simclr = np.array(X_simclr)
    print('simclr rep done...')
    return X_simclr
