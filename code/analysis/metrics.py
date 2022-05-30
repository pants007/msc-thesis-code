
import scipy.stats as st
import torch
import collections.abc
import numpy as np
import analysis.utility as util

def get_all_metrics(model, test_loader, device, taxa):
    model.eval()
    idxs = test_loader.dataset.dataset.indices
    all_metrics = []
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            labels = torch.stack(labels).to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            if not isinstance(outputs, collections.abc.Sequence):
                outputs = [outputs]

            metrics_dict = {}
            for i in range(len(taxa)):
                metrics_dict[taxa[i] + '_logits'] = outputs[i].squeeze().cpu().tolist()
                metrics_dict[taxa[i] + '_label'] = labels[i].item()
            metrics_dict['path'] = test_loader.dataset.dataset.dataset._get_path(idxs[idx])
            all_metrics.append(metrics_dict)

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


def k_fold(corrects, k,  seed=0):
    np.random.seed(seed)
    np.random.shuffle(corrects)
    array_list = np.split(corrects, k)
    acc_list = np.empty(k)
    for i in range(k):
        partition = array_list[:i] + array_list[i+1:]
        partition_corrects = 0
        partition_elems = 0
        for array in partition:
            partition_corrects += array.sum()
            partition_elems += len(array)
        acc_list[i] = (partition_corrects / partition_elems)
    return acc_list


def k_fold_stats(models_info, parent_index, index, k, discard_n=3, seed=0, alpha=0.95):
    dataframe = util.calc_dataframe(models_info, parent_index, index)
    logits = np.array(dataframe['test_set']['species_logits'].to_list())
    preds = np.argmax(logits, 1)
    labels = np.array(dataframe['test_set']['species_label'].to_list())
    corrects = preds == labels
    accs = k_fold(corrects[:-discard_n], k, seed)
    acc_mean = np.mean(accs)
    acc_std_0 = np.std(accs)
    acc_std_1 = st.sem(accs)
    acc_conf_interval = st.norm.interval(
        alpha=alpha, loc=acc_mean, scale=acc_std_1)
    return acc_mean, acc_std_0, acc_conf_interval
