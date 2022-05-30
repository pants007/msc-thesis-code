import os
import ssl
import random

import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
from torchvision.datasets.utils import download_url, extract_archive
from torch.utils.data import Dataset, DataLoader, Subset

def download_dataset(url: str = 'https://sid.erda.dk/share_redirect/heaAFNnmaG/data.zip',
                     zip_name: str = 'beetles.zip', folder_name: str = 'beetles',
                     force_download: bool = False, root: str = './data/') -> str:
    ssl._create_default_https_context = ssl._create_unverified_context
    archive = os.path.join(root, zip_name)
    data_folder = os.path.join(root, folder_name)
    if (not os.path.exists(data_folder) or force_download):
        download_url(url, root, zip_name)
        extract_archive(archive, data_folder, False)
    return data_folder

def generate_taxonomy_dataframe(csv_in_path='data/beetles/taxonomy.csv', csv_out_path='data/beetles/taxonomy-modified.csv', drop_min = None):
    dataframe = pd.read_csv(csv_in_path)
    if drop_min:
        samples_per_species = dataframe['species'].value_counts()
        species_to_drop = samples_per_species.where(samples_per_species <= drop_min).dropna().axes[0].to_list()
        for species in species_to_drop:
            dataframe = dataframe[dataframe['species'] != species]
        dataframe.to_csv(csv_out_path, index=False)
    return dataframe


def generate_labels_dict(dataframe, taxa=['subfamily', 'tribe', 'genus', 'species']):
    labels_dict = {}
    for taxon in taxa:
        labels = sorted(dataframe[taxon].unique())
        labels_dict[taxon] = {}
        for idx, label in enumerate(labels):
            labels_dict[taxon][label] = idx
    return labels_dict

class BeetleSet(Dataset):
    def __init__(self, taxa=['subfamily', 'tribe', 'genus', 'species'], images_path ='data/beetles/', csv_path='data/beetles/taxonomy.csv'):
        self.dataframe = pd.read_csv(csv_path)
        self.taxa = taxa
        self.images_path = images_path
        self.labels_dict = generate_labels_dict(self.dataframe, taxa=taxa)
    def __len__(self):
        return len(self.dataframe)
    
    def _get_path(self, idx):
        species = self.dataframe['species'][idx].replace(' ', '_').lower()
        id = self.dataframe['id'][idx]
        path = self.images_path + 'images/' + species + '/' + id + '.jpg'
        return path

    def __getitem__(self, idx):
        labels = []
        for taxon in self.taxa:
            label = self.labels_dict[taxon][self.dataframe[taxon][idx]]
            labels.append(label)
        path = self._get_path(idx)
        img = Image.open(path)
        return img, labels # TODO could consider using a dictionary for labels

class TransformsDataset(Dataset):
    '''Transforms a Dataset by a given transformation'''

    def __init__(self, dataset, transform  = None, label_transform = None): #TODO should be called dataset
        self.dataset = dataset
        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.label_transform is not None:
            y = self.label_transform(y)
        return x, y

    def __len__(self):
        return len(self.dataset)

def stratified_split(dataset, train_split=0.8, val_split=0.1, seed = None):
    if seed is not None:
        np.random.seed(seed)

    assert train_split + val_split < 1 #ensure that the split makes sense

    train_idxs = []
    val_idxs = []
    test_idxs = []

    groups = dataset.dataframe.groupby('species').groups
    for group in groups:
        group_arr = groups[group].to_numpy()
        np.random.shuffle(group_arr)
        num_idxs = len(group_arr)
        num_train_idxs = int(num_idxs * train_split)
        num_val_idxs = int(num_idxs * val_split + 0.5)
        num_test_idxs = int(num_idxs - (num_train_idxs + num_val_idxs))

        train_idx_arr = group_arr[0:num_train_idxs]
        val_idx_arr = group_arr[num_train_idxs:num_train_idxs+num_val_idxs]
        test_idx_arr = group_arr[num_train_idxs+num_val_idxs:num_idxs]

        train_idxs = train_idxs + list(train_idx_arr)
        val_idxs = val_idxs + list(val_idx_arr)
        test_idxs = test_idxs + list(test_idx_arr)

        #verify that the sum of the index splits is equal to the total amount of indices
        assert num_train_idxs + num_val_idxs + num_test_idxs == num_idxs

        #verify that each array of indices contains the correct amount of indices
        assert len(train_idx_arr) == num_train_idxs
        assert len(val_idx_arr) == num_val_idxs
        assert len(test_idx_arr) == num_test_idxs
        assert len(train_idx_arr) > 0
        assert len(val_idx_arr) > 0
        assert len(test_idx_arr) > 0
        #verify that no indices are repeated
        assert len(np.intersect1d(train_idx_arr, val_idx_arr)) == 0
        assert len(np.intersect1d(train_idx_arr, test_idx_arr)) == 0
        assert len(np.intersect1d(test_idx_arr, val_idx_arr)) == 0

    return train_idxs, val_idxs, test_idxs

def stratified_split_equal(dataset, split=0.2, seed = None):
    if seed is not None:
        np.random.seed(seed)

    assert split < 1 #ensure that the split makes sense

    train_idxs = []
    val_idxs = []
    test_idxs = []

    groups = dataset.dataframe.groupby('species').groups
    for group in groups:
        group_arr = groups[group].to_numpy()

        np.random.shuffle(group_arr)
        num_idxs = len(group_arr)
        num_val_idxs = int(num_idxs * (split/2.0) + 0.5) #0.1 split
        num_train_idxs = num_idxs - num_val_idxs*2 #1.0 - 0.1 * 2 = 0.8 ergo test = 0.1

        train_idx_arr = group_arr[0:num_train_idxs]
        val_idx_arr = group_arr[num_train_idxs:num_train_idxs+num_val_idxs]
        test_idx_arr = group_arr[num_train_idxs+num_val_idxs:num_idxs]

        train_idxs = train_idxs + list(train_idx_arr)
        val_idxs = val_idxs + list(val_idx_arr)
        test_idxs = test_idxs + list(test_idx_arr)

        #verify that the sum of the index splits is equal to the total amount of indices
        assert num_train_idxs + num_val_idxs*2 == num_idxs

        #verify that each array of indices contains the correct amount of indices
        assert len(train_idx_arr) == num_train_idxs
        assert len(val_idx_arr) == num_val_idxs
        assert len(test_idx_arr) == num_val_idxs
        assert len(train_idx_arr) > 0
        assert len(val_idx_arr) > 0
        assert len(test_idx_arr) > 0

        #verify that no indices are repeated
        assert len(np.intersect1d(train_idx_arr, val_idx_arr)) == 0
        assert len(np.intersect1d(train_idx_arr, test_idx_arr)) == 0
        assert len(np.intersect1d(test_idx_arr, val_idx_arr)) == 0
        
    return train_idxs, val_idxs, test_idxs

def test_stratified_splitting(beetleset):
    train_idxs, val_idxs, test_idxs = stratified_split(beetleset)

    hist_array = np.empty((4,len(train_idxs)))
    for i,j in enumerate(train_idxs):
        _, labels = beetleset[j]
        hist_array[0,i] = labels['subfamily']
        hist_array[1,i] = labels['tribe']
        hist_array[2,i] = labels['genus']
        hist_array[3,i] = labels['species']

    hist_array2 = np.empty((4, len(val_idxs)))
    for i, j in enumerate(val_idxs):
        _, labels = beetleset[j]
        hist_array2[0, i] = labels['subfamily']
        hist_array2[1, i] = labels['tribe']
        hist_array2[2, i] = labels['genus']
        hist_array2[3, i] = labels['species']


    hist_array3 = np.empty((4, len(test_idxs)))
    for i, j in enumerate(test_idxs):
        _, labels = beetleset[j]
        hist_array3[0, i] = labels['subfamily']
        hist_array3[1, i] = labels['tribe']
        hist_array3[2, i] = labels['genus']
        hist_array3[3, i] = labels['species']


    print('probability distribution for subfamily')
    print(np.histogram(hist_array[0], density=True, bins=[0, 1, 2])[0])
    print(np.histogram(hist_array2[0], density=True, bins=[0, 1, 2])[0])
    print(np.histogram(hist_array3[0], density=True, bins=[0, 1, 2])[0])
    print('probability distribution for tribe')
    print(np.histogram(hist_array[1], density=True, bins=np.arange(5))[0])
    print(np.histogram(hist_array2[1], density=True, bins=np.arange(5))[0])
    print(np.histogram(hist_array3[1], density=True, bins=np.arange(5))[0])
    print('probability distribution for genus')
    print(np.histogram(hist_array[2], density=True, bins=np.arange(45))[0])
    print(np.histogram(hist_array2[2], density=True, bins=np.arange(45))[0])
    print(np.histogram(hist_array3[2], density=True, bins=np.arange(45))[0])
    print('probability distribution for species')
    print(np.histogram(hist_array[3], density=True, bins=np.arange(198))[0])
    print(np.histogram(hist_array2[3], density=True, bins=np.arange(198))[0])
    print(np.histogram(hist_array3[3], density=True, bins=np.arange(198))[0])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(dataset, split, transforms, label_transform, batch_size, 
                    num_workers, seed, dataloader_seed = None, pin_memory = True):
    assert len(transforms) == 3
    train_idxs, val_idxs, test_idxs = stratified_split_equal(dataset, split, seed)
    train_set = TransformsDataset(
        Subset(dataset, train_idxs), transforms[0],label_transform)
    val_set = TransformsDataset(
        Subset(dataset, val_idxs), transforms[1], label_transform)
    test_set = TransformsDataset(
        Subset(dataset, test_idxs), transforms[2], label_transform)
    dataset_sizes = {}
    dataset_sizes['train'] = len(train_set)
    dataset_sizes['val'] = len(val_set)
    dataset_sizes['test'] = len(test_set)

    dataloaders = {}
    f = None
    g = None
    if dataloader_seed is not None:
        f = seed_worker
        g = torch.Generator()
        g.manual_seed(dataloader_seed)
        
    dataloaders['train'] = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        pin_memory= pin_memory, generator = g, worker_init_fn= f)
    dataloaders['val'] = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
        pin_memory= pin_memory, generator = g, worker_init_fn= f)
    dataloaders['test'] = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
        pin_memory= pin_memory, generator = g, worker_init_fn= f)
    return dataloaders, dataset_sizes

def get_class_freqs(taxa, dataframe, idxs):
    class_frequencies = []
    subframe = dataframe.iloc[idxs]
    for taxon in taxa:
        grouped = subframe.groupby(taxon)
        class_freq = torch.tensor(grouped.size().to_list())
        class_frequencies.append(class_freq)
    return class_frequencies

def get_class_freqs_dict(taxa, dataframe, idxs):
    class_frequencies = {}
    subframe = dataframe.iloc[idxs]
    for taxon in taxa:
        grouped = subframe.groupby(taxon)
        class_freq = grouped.size().to_dict()
        class_frequencies[taxon] = class_freq
    return class_frequencies