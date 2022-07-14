import numpy as np
import matplotlib.pyplots as plt
from get_feat import get_features
from helpers import rebin_nd, match_cluster_labels
from parallel_help import load_pkl
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap

input_shape = (32, 32, 5)

manga_normed = load_pkl('manga_normed_sigma_clip_public.pkl')
manga_normed_bin = np.zeros((manga_normed.shape[0], input_shape[0],\
					 input_shape[1], input_shape[2]), dtype=np.float32)
for ii in range(manga_normed.shape[0]):
    manga_normed_bin[ii, :, :, :] = rebin_nd(manga_normed[ii], [32, 32, 1])

n_zero_pix = np.sum(np.where(manga_normed_bin==0., 1, 0), axis=0)


pca = PCA(n_components=10)
X_pca = pca.fit_transform(np.reshape(manga_normed_bin, (manga_normed_bin.shape[0], -1)))
scaler = StandardScaler()
X_pca = scaler.fit_transform(X_pca)

model_1 = 'model_'
model_2 = 'model_'

X_simclr_1 = get_features(model_1, manga_normed_bin)
X_simclr_2 = get_features(model_2, manga_normed_bin)

scaler = Normalizer()
X_simclr_1 = scaler.fit_transform(X_simclr_1)
X_simclr_2 = scaler.fit_transform(X_simclr_2)


total_clusters = 3
cluster_simclr_1 = KMeans(n_clusters=total_clusters, seed=0).fit(X_simclr_1)
labels_simclr_1 = cluster_simclr_1.labels_
cluster_simclr_2 = KMeans(n_clusters=total_clusters, seed=0).fit(X_simclr_2)
labels_simclr_2 = cluster_simclr_2.labels_








