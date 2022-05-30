import os
import numpy as np
import matplotlib.pyplot as plt
import analysis.utility as util
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable

cluster_path = './../output/msc-thesis-22/clusters/'

def get_clusters(features, labels, method='label'):
    cluster_labels = np.unique(labels)
    label_cluster = np.zeros((cluster_labels.shape[0], features.shape[1]))
    if method == 'label':
        for i in range(label_cluster.shape[0]):
            idx = np.where(labels == cluster_labels[i])[0]
            label_cluster[i] = np.mean(features[idx], axis=0)
        return label_cluster
    else:
        kmeans = KMeans(n_clusters=cluster_labels.shape[0])
        kmeans.fit(features)

        return kmeans.cluster_centers_

def get_knn_clusters(dataframe, taxa):
    features = np.array(dataframe[taxa + '_logits'].to_list())
    labels = dataframe[taxa + '_label'].to_numpy()
    cluster_labels = np.unique(labels)
    kmeans = KMeans(n_clusters=cluster_labels.shape[0])
    kmeans.fit(features)
    return kmeans.cluster_centers_

def get_label_clusters(dataframe, taxa):
    features = np.array(dataframe[taxa + '_logits'].to_list())
    labels = dataframe[taxa + '_label'].to_numpy()
    cluster_labels = np.unique(labels)
    label_cluster = np.zeros((cluster_labels.shape[0], features.shape[1]))
    for i in range(label_cluster.shape[0]):
        idx = np.where(labels == cluster_labels[i])[0]
        label_cluster[i] = np.mean(features[idx], axis=0)
    return label_cluster

def closest_cluster(features, clusters):
    ret = np.zeros((features.shape[0], clusters.shape[0]))
    for i in range(clusters.shape[0]):
        ret[:,i] = np.linalg.norm(features - clusters[i], axis=1)

    return np.argmin(ret,axis=1)

def clusters_dists(c_1, c_2):
    ret = np.zeros((c_1.shape[0], c_2.shape[0]))
    for i in range(c_2.shape[0]):
        ret[:,i] = np.linalg.norm(c_1 - c_2[i], axis=1)

    return ret

def plot_genus_lines(ax, dataframe):
    classes = dataframe[['genus_label','species_label']].drop_duplicates()
    classes = np.bincount(classes['genus_label'].to_numpy())
    for i in range(classes.shape[0]):
        start = np.sum(classes[:i])-0.5
        ax.plot([-0.5,196.5], [start,start], c='r')

def plot_genus_lines_(ax, data):
    for i in range(data.shape[0]):
        start = np.sum(data[:i])-0.5
        ax.plot([-0.5,196.5], [start,start], c='r')

def cluster_dist_image(model, comp_type, taxa, species_lines=False, dataset_type='train_set', sp_np=None):
    if not util.is_dataset(dataset_type) or not util.is_taxa(taxa):
        return
    if comp_type not in ['label', 'knn', 'cross']:
        print('comp_type has to be label, knn or cross')
        return 

    dataframe = model[dataset_type]
    if comp_type == 'label':
        label_clusters = get_label_clusters(dataframe, taxa)
        dif = clusters_dists(label_clusters, label_clusters)
    elif comp_type == 'knn':
        knn_clusters = get_knn_clusters(dataframe, taxa)
        dif = clusters_dists(knn_clusters, knn_clusters)
    else:
        label_clusters = get_label_clusters(dataframe, taxa)
        knn_clusters = get_knn_clusters(dataframe, taxa)
        dif = clusters_dists(label_clusters, knn_clusters)
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    im = ax.imshow(dif)
    f_l = ''
    if species_lines:
        if sp_np is None:
            plot_genus_lines(ax, dataframe)
        else:
            plot_genus_lines_(ax, sp_np)
        f_l = '_lines'
    if not os.path.exists(cluster_path + model['model_name'] + '/'):
        os.mkdir(cluster_path + model['model_name'] + '/')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)
    plt.tight_layout()
    fig.savefig(cluster_path + model['model_name'] + '/' + comp_type + '_' + taxa + f_l +'_clusterdist.png')


def cluster_std(model, taxa,  cluster_method, dataset_type='train_set'):
    if not util.is_dataset(dataset_type) or not util.is_taxa(taxa):
        return
    if cluster_method not in ['label', 'knn']:
        print('cluster_method has to be label or nnn')
        return
    dataframe = model['train_set']
    if cluster_method == 'knn':
        clusters = get_knn_clusters(dataframe, taxa)
    else:
        clusters = get_label_clusters(dataframe, taxa)
    features = np.array(model[dataset_type][taxa+'_logits'].to_list())
    dist_all = clusters_dists(features, clusters)
    cluster_assignment = np.argmin(dist_all, axis=1)
    dist_assigned = dist_all[np.arange(dist_all.shape[0]), cluster_assignment]
    ret = np.zeros((dist_all.shape[1]))
    for i in range(197):
        ret[i] = np.std(dist_assigned[np.where(cluster_assignment==i)[0]])
    return ret