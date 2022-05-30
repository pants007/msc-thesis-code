from dataset import download_dataset, generate_taxonomy_dataframe, BeetleSet, get_dataloaders, get_class_freqs_dict
from loss_functions import WeightedCCELoss, WeightedBCELoss
from train import weighted_fit_tb_recon, weighted_test_recon
from models import SequentialHierarchicalModel, FusionModel, SequentialMultiTaskModel
import fusion_model_dictionaries as fmd
import models
import torch
from torchvision.transforms import Compose, Resize, RandomVerticalFlip, RandomRotation, Normalize, ToTensor
import torchvision
from transforms import RandomResizedCrop
import numpy as np
import matplotlib.pyplot as plt

import json
from pathlib import Path
import pandas as pd

import random
import os

def weighted_pipeline_tb_recon(model_constructor, aux_model_args, model_name, taxa, transforms, seed, batch_size,
                                num_workers, device, num_epochs, loss_coefs=None, root='../output/msc-thesis-22/models/', criterion_constructor=WeightedCCELoss, early_stopping=None, dataloader_seed=None, 
                                optimizer_constructor = torch.optim.Adam,  aux_optimizer_args = {}, 
                                scheduler_constructor = torch.optim.lr_scheduler.ExponentialLR, 
                                aux_scheduler_args = {'gamma': 0.995}, reduction='mean', weight_scheme=None, 
                                aux_criterion_args={}, use_amp = True, use_scaler=True, pin_memory = True, 
                                grad_clip_val = 1e+8, load_state_dict=False, tb_root = '../output/msc-thesis-22/tensorboard/',
                                disc_constructor = None, aux_disc_args = {}, optim_d_constructor = torch.optim.Adam,
                                aux_optim_d_args={}, optim_d_scheduler_constructor=torch.optim.lr_scheduler.ExponentialLR, 
                                aux_optim_d_scheduler_args={'gamma': 0.995}, disc_loss_threshold = 0.1, scale_factor = None):
    download_dataset()

    tmap = {'subfamily': '0', 'tribe': '1', 'genus': '2', 'species': '3'}
    tags = model_name.split(' ')
    model_name_ext = tags[0] + '_' + \
        ''.join([tmap[taxon] for taxon in taxa])
    tags[0] = model_name_ext
    model_name_ext = ' '.join(tags)

    
    Path(root + model_name_ext).mkdir(parents=True, exist_ok=True)

    generate_taxonomy_dataframe('data/beetles/taxonomy.csv', root + model_name_ext +
                                '/taxonomy-modified.csv', drop_min=9)
    dataset = BeetleSet(csv_path=root + model_name_ext +
                        '/taxonomy-modified.csv', taxa=taxa)
    dataloaders, dataset_sizes = get_dataloaders(dataset, 0.2, transforms, None, batch_size,
                                                 num_workers, seed, dataloader_seed, pin_memory)
    if weight_scheme == 'icf':
        train_idxs = dataloaders['train'].dataset.dataset.indices
        class_freqs_dict = get_class_freqs_dict(
            taxa, dataset.dataframe, train_idxs)
        class_freqs = []
        for taxon in taxa:
            keys, vals = zip(*class_freqs_dict[taxon].items())
            assert list(keys) == list(dataset.labels_dict[taxon].keys())
            class_freqs.append(torch.tensor(list(vals)))
        weights_per_taxon = [1.0/taxon_freqs for taxon_freqs in class_freqs]
    else:
        weights_per_taxon = None

    if weights_per_taxon is not None:
        assert len(weights_per_taxon) == len(taxa)
        for i, taxon in enumerate(taxa):
            assert torch.is_tensor(weights_per_taxon[i])
            assert len(weights_per_taxon[i].shape) == 1
            assert len(weights_per_taxon[i]) == len(dataset.labels_dict[taxon])

    if loss_coefs is not None:
        assert torch.is_tensor(loss_coefs)
        assert len(loss_coefs.shape) == 1
        if disc_constructor is None:
            assert len(loss_coefs) == len(taxa) + 1
        else:
            assert len(loss_coefs) == len(taxa) + 2
    else:
        if disc_constructor is None:
            loss_coefs = torch.ones(len(taxa) + 1)
        else:
            loss_coefs = torch.ones(len(taxa) + 2)

    classes_per_taxon = [len(dataset.labels_dict[taxon]) for taxon in taxa]
    constructor_args = {**aux_model_args, **
                        {'classes_per_taxon': classes_per_taxon}}
    model = model_constructor(**constructor_args).to(device)
    #print(model)
    criterion_args = {**aux_criterion_args, **
                      {'weights_per_taxon': weights_per_taxon}}
    criterion = criterion_constructor(**criterion_args)
    
    optimizer_args = {**aux_optimizer_args, **
                      {'params': model.parameters()}}
    optimizer = optimizer_constructor(**optimizer_args)
    
    scheduler_args = {**aux_scheduler_args, **
                    {'optimizer': optimizer}}
    
    scheduler = scheduler_constructor(**scheduler_args)
    
    disc = None
    optim_d = None
    optim_d_scheduler = None
    if disc_constructor is not None:
        disc_args = {**aux_disc_args}
        disc = disc_constructor(**disc_args).to(device)
        optim_d_args = {**aux_optim_d_args, **
                        {'params': disc.parameters()}}
        optim_d = optim_d_constructor(**optim_d_args)
        optim_d_scheduler_args = {**aux_optim_d_scheduler_args, **
                                  {'optimizer': optim_d}}
        optim_d_scheduler = optim_d_scheduler_constructor(**optim_d_scheduler_args)
    
    if __name__ == '__main__' and not load_state_dict:
            model, metrics = weighted_fit_tb_recon(model, criterion, optimizer, scheduler, dataloaders, taxa, dataset_sizes,
                                                    device, loss_coefs, num_epochs, early_stopping=early_stopping, reduction=reduction, use_amp=use_amp, use_scaler=use_scaler, grad_clip_val = grad_clip_val, 
                                                    tb_path=tb_root, tags=tags, disc=disc, optim_d=optim_d, 
                                                    optim_d_scheduler = optim_d_scheduler,  disc_loss_threshold=disc_loss_threshold,
                                                    scale_factor=scale_factor)
    else:
        state = torch.load(root + model_name_ext + '/state_dict.pt')
        model.load_state_dict(state)
        with open(root + model_name_ext + '/metrics.json') as f:
            metrics = json.load(f)
        
    torch.save(model.state_dict(), root + model_name_ext + '/state_dict.pt')
    criterion.weights_per_taxon = None
    test_metrics = weighted_test_recon(model, criterion, dataloaders['test'], device,
                                 dataset_sizes, taxa, tb_root, tags, disc=disc, scale_factor=scale_factor)
    for taxon in taxa + ['total']:
        metrics[taxon]['test'] = test_metrics[taxon]
    metrics['test_recon_loss'] = test_metrics['recon_loss']
    metrics['test_adv_loss'] = test_metrics['adv_loss']
    metrics['test_disc_loss'] = test_metrics['disc_loss']

    metrics['model_type'] = str(type(model))
    metrics['classes_per_taxon'] = classes_per_taxon
    metrics['aux_model_args'] = {}

    for name, arg in aux_model_args.items():
        if name == 'hidden_features_per_taxon':
            metrics['aux_model_args'][name] = arg
        elif name == 'model':
            metrics['aux_model_args'][name] = str(type(arg))
        elif type(arg) == dict:
            metrics['aux_model_args'][name] = arg
        else:
            metrics['aux_model_args'][name] = str(arg)

    metrics['criterion_type'] = str(type(criterion))
    metrics['reduction'] = reduction
    
    metrics['weight_scheme'] = weight_scheme
    if weights_per_taxon is None:
        metrics['weights_per_taxon'] = weights_per_taxon
    else:
        metrics['weights_per_taxon'] = [tensor.tolist()
                                        for tensor in weights_per_taxon]
     
    metrics['aux_criterion_args'] = {}   
    for name, arg in aux_criterion_args.items():
        metrics['aux_criterion_args'][name] = arg
    
    metrics['optimizer_type'] = str(type(optimizer))
    metrics['aux_optimizer_args'] = {}
    for name, arg in aux_optimizer_args.items():
        metrics['aux_optimizer_args'][name] = arg
    
    metrics['scheduler_type'] = str(type(scheduler))
    metrics['aux_scheduler_args'] = {}
    for name, arg in aux_scheduler_args.items():
        if name == 'lr_lambda' :
            metrics['aux_scheduler_args'][name] = str(arg)
        else:       
            metrics['aux_scheduler_args'][name] = arg
    #TODO we could probably simplify the boiler plate code below as it basically a copy of what is done above       
    if optim_d is not None:
        metrics['optim_d_type'] = str(type(optim_d))
        metrics['aux_optim_d_args'] = {}
        for name, arg in aux_optim_d_args.items():
            metrics['aux_optim_d_args'][name] = arg
        
        metrics['optim_d_scheduler_type'] = str(type(optim_d_scheduler))
        metrics['aux_optim_d_scheduler_args'] = {}
        for name, arg in aux_optim_d_scheduler_args.items():
            if name == 'lr_lambda' :
                metrics['aux_optim_d_scheduler_args'] = str(arg)
            else:
                metrics['aux_optim_d_scheduler_args'] = arg

    metrics['batch_size'] = batch_size

    metrics['seed'] = seed

    with open(root + model_name_ext + '/metrics.json', 'w') as file:
        json.dump(metrics, file, indent=4)

    return model, metrics

def seed_global(seed: int):
    #os.environ['PYTHONHASHSEED'] = str(seed) i am not sure about this one
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #force pytorch to use deterministic algorithms for all operations when available,
    # and throw and error when operations cannot be executed deterministically.
    #torch.use_deterministic_algorithms(True)


def reset_seed(seed, resave=True):
    seed_global(seed)
    base_model = torchvision.models.resnet18(False)
    if os.path.exists(models_root + 'init_weights.pt') and not(resave):
        init_weights = torch.load(models_root + 'init_weights.pt')
        base_model.load_state_dict(init_weights)
    else:
        init_weights = base_model.state_dict()
        torch.save(init_weights, models_root + 'init_weights.pt')
    return base_model

models_root = '../output/msc-thesis-22/models/'
baseline_root = models_root + 'baseline/'
extensions_root = models_root + 'extensions/'
custom_root = models_root + 'custom/'
relics_root = models_root + 'relics/'
taxa = ['subfamily', 'tribe', 'genus', 'species']
taxa_rev = taxa[::-1]
BEETLENET_MEAN = np.array(
    [0.8442649, 0.82529384, 0.82333773], dtype=np.float32)
BEETLENET_STD = np.array([0.28980458, 0.32252666, 0.3240354], dtype=np.float32)
BEETLENET_AVERAGE_SHAPE = (224, 448)
default_transforms = [Compose([Resize(BEETLENET_AVERAGE_SHAPE), ToTensor(
), Normalize(BEETLENET_MEAN, BEETLENET_STD)])] * 3
batch_size = 32
num_workers = 6
num_epochs = 400
seed = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
train_transforms = Compose([
    RandomVerticalFlip(p=0.5),
    RandomRotation((-3, 3), fill=255),
    RandomResizedCrop(min_scale=0.95, max_scale=1),
    Resize(BEETLENET_AVERAGE_SHAPE),
    ToTensor(),
    Normalize(BEETLENET_MEAN, BEETLENET_STD)])
modified_transforms = [train_transforms] + default_transforms[1:]

#-------------- JENS ----------------------

# model_name = 'fusion_forward recon no_disc explicit_encoder decoder_bidirectional'
# model_constructor = models.FusionModel
# #base_model = torchvision.models.resnet18(False)
# aux_model_args = {
#     'graph_dict': fmd.make_forward_graph_dict_recon_3('conv_per_output')}
# #aux_scheduler_args = {'gamma': 1.0}

# model, metrics = weighted_pipeline_tb_recon(model_constructor, aux_model_args, model_name, ['genus', 'species'], modified_transforms,
#                                             seed, batch_size, num_workers, device, num_epochs, loss_coefs=None,
#                                             root='../output/msc-thesis-22/models/custom/', criterion_constructor=WeightedBCELoss,
#                                             reduction='sum', scale_factor=0.5, load_state_dict=False)
# model.to('cpu')
# del model
# torch.cuda.empty_cache()

model_name = 'fusion_forward recon no_disc implicit_encoder spatial'
model_constructor = models.FusionModel
#base_model = torchvision.models.resnet18(False)
aux_model_args = {
    'graph_dict': fmd.make_forward_graph_dict_recon_1b_spatial('conv_per_output')}
#aux_scheduler_args = {'gamma': 1.0}

model, metrics = weighted_pipeline_tb_recon(model_constructor, aux_model_args, model_name, taxa, modified_transforms,
                                            seed, batch_size, num_workers, device, num_epochs, loss_coefs=None,
                                            root='../output/msc-thesis-22/models/custom/', criterion_constructor=WeightedBCELoss,
                                            reduction='sum', scale_factor=0.5, load_state_dict=False)
model.to('cpu')
del model
torch.cuda.empty_cache()


model_name = 'fusion_forward recon no_disc explicit_encoder flat decoder_bidirectional'
model_constructor = models.FusionModel
#base_model = torchvision.models.resnet18(False)
aux_model_args = {
    'graph_dict': fmd.make_forward_graph_dict_recon_3b('conv_per_output')}
#aux_scheduler_args = {'gamma': 1.0}

model, metrics = weighted_pipeline_tb_recon(model_constructor, aux_model_args, model_name, ['genus', 'species'], modified_transforms,
                                            seed, batch_size, num_workers, device, num_epochs, loss_coefs=None,
                                            root='../output/msc-thesis-22/models/custom/', criterion_constructor=WeightedBCELoss,
                                            reduction='sum', scale_factor=0.5, load_state_dict=False)
model.to('cpu')
del model
torch.cuda.empty_cache()
# -------------- MATHIAS ----------------------

# model_name = 'fusion_forward recon no_disc implicit_encoder flat'
# model_constructor = models.FusionModel
# #base_model = torchvision.models.resnet18(False)
# aux_model_args = {
#     'graph_dict': fmd.make_forward_graph_dict_recon_1b_flat('conv_per_output')}
# #aux_scheduler_args = {'gamma': 1.0}

# model, metrics = weighted_pipeline_tb_recon(model_constructor, aux_model_args, model_name, taxa, modified_transforms,
#                                             seed, batch_size, num_workers, device, num_epochs, loss_coefs=None,
#                                             root='../output/msc-thesis-22/models/custom/', criterion_constructor=WeightedBCELoss,
#                                             reduction='sum', scale_factor=0.5, load_state_dict=False)
# model.to('cpu')
# del model
# torch.cuda.empty_cache()

# model_name = 'fusion_forward recon no_disc explicit_encoder spatial'
# model_constructor = models.FusionModel
# #base_model = torchvision.models.resnet18(False)
# aux_model_args = {
#     'graph_dict': fmd.make_forward_graph_dict_reconstruction_1a('conv_per_output', True)}
# #aux_scheduler_args = {'gamma': 1.0}

# model, metrics = weighted_pipeline_tb_recon(model_constructor, aux_model_args, model_name, ['genus', 'species'], modified_transforms,
#                                             seed, batch_size, num_workers, device, num_epochs, loss_coefs=None,
#                                             root='../output/msc-thesis-22/models/custom/', criterion_constructor=WeightedBCELoss,
#                                             reduction='sum', scale_factor=0.5, load_state_dict=False)
# model.to('cpu')
# del model
# torch.cuda.empty_cache()

# model_name = 'fusion_forward recon no_disc explicit_encoder flat fc_preprocess'
# model_constructor = models.FusionModel
# #base_model = torchvision.models.resnet18(False)
# aux_model_args = {
#     'graph_dict': fmd.fusion_forward_recon_flat_decoding('conv_per_output', 'FCPreprocess')}
# #aux_scheduler_args = {'gamma': 1.0}

# model, metrics = weighted_pipeline_tb_recon(model_constructor, aux_model_args, model_name, ['genus', 'species'], modified_transforms,
#                                             seed, batch_size, num_workers, device, num_epochs, loss_coefs=None,
#                                             root='../output/msc-thesis-22/models/custom/', criterion_constructor=WeightedBCELoss,
#                                             reduction='sum', scale_factor=0.5, load_state_dict=False)