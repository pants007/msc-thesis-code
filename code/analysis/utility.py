import torch
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import torchvision
import numpy as np
import pandas as pd
import json
from pathlib import Path
import os
import collections.abc
import matplotlib.pyplot as plt

import models
from dataset import download_dataset, generate_taxonomy_dataframe, BeetleSet, TransformsDataset, generate_labels_dict, stratified_split_equal


root = '../output/msc-thesis-22/models/'

dataframe_path = './dataframes/'
if not os.path.exists(dataframe_path):
    os.mkdir(dataframe_path)


BEETLENET_MEAN = np.array([0.8442649, 0.82529384, 0.82333773], dtype=np.float32)
BEETLENET_STD = np.array([0.28980458, 0.32252666, 0.3240354], dtype=np.float32)
BEETLENET_AVERAGE_SHAPE = (224, 448)
default_transforms = Compose([Resize(BEETLENET_AVERAGE_SHAPE), ToTensor(), Normalize(BEETLENET_MEAN, BEETLENET_STD)])
num_workers = 4
seed = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def load_models_info(root):
    parent_folders_ = [f for f in os.listdir(root) if os.path.isdir(root + f)]
    parent_folders_.sort()
    parent_folders = []

    for f in parent_folders_:
        if f != 'relics':
            parent_folders.append(f)
    
    ret = {}
    for i in range(len(parent_folders)):
        name_based = {}
        subfolder_ = os.listdir(root  + parent_folders[i])
        subfolder_.sort()
        subfolder = []
        for j in subfolder_:
            current_folder = root  + parent_folders[i] + '/' + j + '/'
            if os.path.exists(current_folder + 'taxonomy-modified.csv') and os.path.exists(current_folder + 'state_dict.pt') and os.path.exists(current_folder+'metrics.json') :
                subfolder.append(j)
        for j in range(len(subfolder)):
            current_folder = root  + parent_folders[i] + '/' + subfolder[j] + '/'
            model_dict = {}
            model_dict['taxonomy'] = current_folder + 'taxonomy-modified.csv'
            model_dict['state_dict'] = current_folder + 'state_dict.pt'
            model_dict['model_group'] = parent_folders[i]
            model_dict['model_name'] = subfolder[j]
            model_dict['parent_idx'] = i
            model_dict['idx'] = j

            with open(current_folder+'metrics.json') as jsonFile:
                metrics = json.load(jsonFile)

            keys = list(metrics.keys())
            model_dict['ref_model'] = False
            if 'aux_model_args' in metrics.keys():
                extra_param = metrics['aux_model_args']
                params = []
                for key in extra_param.keys():
                    if key != 'classes_per_taxon' and key != 'model':
                        if type(extra_param[key]) == str and '{' in extra_param[key]:
                            params.append(extra_param[key])
                        else:
                            params.append(extra_param[key])

                    if key == 'model':
                        model_dict['ref_model'] = True
                model_dict['extra_params'] = params
            else:
                model_dict['extra_params'] = []

            # remove checks later
            if 'taxa' in keys:
                model_dict['taxa'] = metrics['taxa']
            else:
                model_dict['taxa'] = []
                
            if 'model_type' in keys:
                model_dict['model_type'] = str(metrics['model_type']).split('.')[1][:-2]
            else:
                model_dict['model_type'] = 'some model'

            name_based[subfolder[j]]=model_dict
        ret[parent_folders[i]] = name_based
    return ret  

def make_info_idx(models_info):
    idx = {}
    parent_keys = list(models_info.keys())
    for i in range(len(parent_keys)):
        sub_dict = {}
        sub_keys = list(models_info[parent_keys[i]].keys())
        for j in range(len(sub_keys)):
            idx_dict = {}
            idx_dict['parent_idx'] = parent_keys[i]
            idx_dict['idx'] = sub_keys[j]
            sub_dict[j] = idx_dict
        idx[i] = sub_dict
    return idx

def show_models(models_info):
    dataframe = pd.DataFrame()
    parent_keys = list(models_info.keys())
    idx = 0
    for i in range(len(parent_keys)):
        sub_keys = list(models_info[parent_keys[i]].keys())
        for j in range(len(sub_keys)):
            current = models_info[parent_keys[i]][sub_keys[j]]
            row = pd.DataFrame({'Group': current['model_group'], 'Model name': current['model_name'],'Model type': current['model_type'],'Parent index': current['parent_idx'],'Index': current['idx'], 'Taxa order' : str(current['taxa'])}, columns=['Group', 'Model name', 'Model type', 'Parent index', 'Index', 'Taxa order'], index = [idx])
            dataframe = pd.concat([dataframe, row], axis = 0)
            idx += 1
    return dataframe

def convert_idx(models_info, group, name):
    if type(group) == int and type(name) == int:
        lookup_idx = make_info_idx(models_info)
        current = lookup_idx[group][name]
        first_idx = current['parent_idx']
        second_idx = current['idx']
        return first_idx, second_idx
    return group, name

def get_all_metrics(model, test_loader, device, taxa):
    model.eval()
    idxs = test_loader.dataset.dataset.indices
    batch_size = test_loader.batch_size
    all_metrics = []
    logits = []
    label = []
    for i in range(len(taxa)):
        logits.append([])
        label.append([])
    paths = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            labels = torch.stack(labels).to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            if not isinstance(outputs, collections.abc.Sequence):
                outputs = [outputs]
            recon = 0
            if outputs[0].dim() > 2:
                recon = 1
                
            for i in range(len(taxa)):
                logits[i] += outputs[i+recon].squeeze().cpu().tolist()
                label[i] += labels[i].cpu().tolist()
            for i in idxs[idx*batch_size:(idx+1)*batch_size]:
                paths.append(test_loader.dataset.dataset.dataset._get_path(i))
    for i in range(len(paths)):
        metrics = {}
        for t in range(len(taxa)):
            metrics[taxa[t] + '_logits'] = logits[t][i]
            metrics[taxa[t] + '_label'] = label[t][i]
        metrics['path'] = paths[i]
        all_metrics.append(metrics)
    return all_metrics


def get_accuracy(dataframe, taxon, class_label):
    grouped = dataframe.groupby(taxon + '_label')
    group = grouped.get_group(class_label)
    preds = group[taxon +'_logits'].transform(lambda x: np.argmax(x))
    corrects = group[taxon + '_label'] == preds
    mean = corrects.mean()
    return group, mean

def get_dists(dataframe, taxon, class_label):
    grouped = dataframe.groupby(taxon + '_label')
    group = grouped.get_group(class_label)
    preds = group[taxon + '_logits'].transform(lambda x: np.argmax(x))
    dists = abs(group[taxon + '_label'] - preds)
    return group, dists

def calc_dataframe(model_info, group, name, transforms=default_transforms, num_workers=num_workers, seed=seed, device=device, models= models, recalc=False):
    fst_idx, snd_idx = convert_idx(model_info, group, name)
    model_dict = model_info[fst_idx][snd_idx]
    if len(model_dict['taxa']) == 1:
        model_name = snd_idx[:-2]
        sub_keys = np.array([y for y in list(model_info[fst_idx].keys()) if y[:-2] == model_name])
        special_model = {'model_name' : model_name + '_combined'}
        for dataset_type in ['train', 'val', 'test']:
            if not os.path.exists(dataframe_path + model_name + '_combined' + '/' + dataset_type + '.csv') and not recalc:
                dataframes = []
                for n in sub_keys:
                    model_dict = model_info[fst_idx][n]
                    dataframes.append(calc_dataframe_(model_dict, dataset_type=dataset_type, transforms=transforms, num_workers=num_workers, seed=seed, device=device, models= models, recalc=recalc))
                
                comb_dataframe = dataframes[0]
                for i in range(1, len(dataframes)):
                    comb_dataframe = pd.merge(comb_dataframe,dataframes[i])
                    
                save_dataframe(comb_dataframe, special_model, dataset_type)
            special_model[dataset_type + '_set'] = read_dataframe(special_model, dataset_type)
        return special_model

    else: 
        model_dict = model_info[fst_idx][snd_idx]
        for dataset_type in ['train', 'val', 'test']:
            model_dict[dataset_type + '_set'] = calc_dataframe_(model_dict, dataset_type=dataset_type, transforms=transforms, num_workers=num_workers, seed=seed, device=device, models= models, recalc=recalc)
            save_dataframe(model_dict[dataset_type + '_set'], model_dict, dataset_type)
        return model_dict

def calc_dataframe_(model_dict, dataset_type, transforms=default_transforms, num_workers=num_workers, seed=seed, device=device, models= models, recalc=False):
    if os.path.exists(dataframe_path + model_dict['model_name'] + '/' + dataset_type + '.csv') and not recalc:
        return read_dataframe(model_dict, dataset_type)
    else:
        download_dataset()
        beetleset = BeetleSet(csv_path=model_dict['taxonomy'], taxa=model_dict['taxa'])
        dataset_idx = stratified_split_equal(beetleset, seed = seed)
        dataset_dict = {'train' : 0, 'val': 1, 'test': 2}

        set = TransformsDataset(Subset(beetleset, dataset_idx[dataset_dict[dataset_type]]), transform=transforms)
        loader = DataLoader(
            set, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory= True)

        labels_per_taxon = [len(beetleset.labels_dict[taxon]) for taxon in model_dict['taxa']]
        model_type = model_dict['model_type']
        extra_params = model_dict['extra_params']
        if model_dict['ref_model']:
            resmodels = [torchvision.models.resnet18, torchvision.models.resnet34, torchvision.models.resnet50]
            for model_ref in resmodels:
                try:
                    model = models.__dict__[model_type](model_ref(False), labels_per_taxon, *extra_params).to(device)
                    model.load_state_dict(torch.load(model_dict['state_dict']))
                    metrics_test = get_all_metrics(model, loader, device, model_dict['taxa'])
                    return pd.DataFrame.from_dict(metrics_test)   
                except:
                    pass
        else:
            model = models.__dict__[model_type](labels_per_taxon, *extra_params).to(device)
            model.load_state_dict(torch.load(model_dict['state_dict']))
            metrics_test = get_all_metrics(model, loader, device, model_dict['taxa'])
            return pd.DataFrame.from_dict(metrics_test)    




def get_same(dfs, taxa):
    ret = dfs[0][['id', taxa, 'path']]
    for i in range(1, len(dfs)):
        ret = pd.merge(ret, dfs[i][['id', taxa, 'path']])
    return ret


def df_correct(dataframe):
    subfamily = dataframe['subfamily_logits'] == dataframe['subfamily_label'] 
    tribe = dataframe['tribe_logits'] == dataframe['tribe_label'] 
    genus = dataframe['genus_logits'] == dataframe['genus_label'] 
    species = dataframe['species_logits'] == dataframe['species_label'] 
    return pd.DataFrame({'id': dataframe['id'], 'subfamily':subfamily, 'tribe':tribe,'genus':genus,'species':species,'path' : dataframe['path']})

def df_pred(dataframe):
    ret = dataframe.copy()
    ret['species_logits'] = ret.species_logits.apply(lambda x: np.argmax(x))
    ret['genus_logits'] = ret.genus_logits.apply(lambda x: np.argmax(x))
    ret['tribe_logits'] = ret.tribe_logits.apply(lambda x: np.argmax(x))
    ret['subfamily_logits'] = ret.subfamily_logits.apply(lambda x: np.argmax(x))
    return ret

def save_dataframe(dataframe, model, set):
    if not os.path.exists(dataframe_path + model['model_name']):
        os.mkdir(dataframe_path + model['model_name'])
    dataframe.to_csv(dataframe_path + model['model_name'] + '/' + set + '.csv')

def read_dataframe(model, set):
    path = dataframe_path + model['model_name'] + '/' + set + '.csv'
    return read_dataframe_(path)

def read_dataframe_(file):
    dataframe = pd.read_csv(file)
    order = ['id']
    keys = list(dataframe.columns)
    if 'subfamily_logits' in keys:
        dataframe['subfamily_logits'] = dataframe.subfamily_logits.apply(lambda x: [float(y) for y in x[1:-1].split(',')])
        order += ['subfamily_logits',	'subfamily_label']
    if 'tribe_logits' in keys:
        dataframe['tribe_logits'] = dataframe.tribe_logits.apply(lambda x: [float(y) for y in x[1:-1].split(',')])
        order += ['tribe_logits',	'tribe_label']
    if 'genus_logits' in keys:
        dataframe['genus_logits'] = dataframe.genus_logits.apply(lambda x: [float(y) for y in x[1:-1].split(',')])
        order += ['genus_logits',	'genus_label']
    if 'species_logits' in keys:
        dataframe['species_logits'] = dataframe.species_logits.apply(lambda x: [float(y) for y in x[1:-1].split(',')])
        order += ['species_logits',	'species_label']
    if 'id' not in keys:
        dataframe.rename(columns = {'Unnamed: 0':'id'},  inplace = True)
    return dataframe[order + ['path']]

def is_taxa(s):
    if s in ['subfamily', 'tribe', 'genus', 'species']:
        return True
    else:
        print('taxa has to be subfamily, tribe, genus or species')
        return False

def is_dataset(s):
    if s in ['train_set', 'val_set', 'test_set']:
        return True
    else:
        print('dataset_type has to be train_set, val_set or test_set')
        return False

def compare_fit(set_a, set_b, norm=False):
    if norm:
        set_a = set_a - np.min(set_a)
        set_a = set_a / np.max(set_a)
        set_b = set_b - np.min(set_b)
        set_b = set_b / np.max(set_b)
    idx_sort = np.argsort(set_b)
    sort_a = set_a[idx_sort]
    sort_b = set_b[idx_sort]
    x = np.arange(sort_a.shape[0])
    pf1 = np.polyfit(x, sort_a, 1)
    pf2 = np.polyfit(x, sort_a, 2)
    fig, ax = plt.subplots(1,1,figsize=(30,10))
    ax.plot(sort_a, label='set_a')
    ax.plot(sort_b, label='set_b')
    ax.plot(pf1[0]*x + pf1[1], label='polyfit_1')
    ax.plot(pf2[0]*(x**2)  + pf2[1]*x + pf2[2], label='polyfit_2')
    ax.legend()
    return fig