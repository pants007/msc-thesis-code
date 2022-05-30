import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, RandomVerticalFlip, RandomRotation, Normalize, ToTensor
import torchvision
import numpy as np
import collections.abc

from train import weighted_loss
from loss_functions import WeightedCCELoss, WeightedBCELoss
import models
import fusion_model_dictionaries as fmd
from transforms import RandomResizedCrop
from dataset import download_dataset, generate_taxonomy_dataframe, BeetleSet, get_dataloaders, get_class_freqs_dict

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

use_amp = True
use_scaler=True
pin_memory = True
grad_clip_val = 1e+8


reduction='sum'
num_workers = 0


model_constructor = models.FusionModel
aux_model_args = {'graph_dict': fmd.make_backward_graph_dict('conv_per_output')}

name = 'profile_backward_cop_32'

wait=1
warmup=1
active=3
repeat=1



taxa = ['subfamily', 'tribe', 'genus', 'species']
taxa_rev = taxa[::-1]
BEETLENET_MEAN = np.array(
    [0.8442649, 0.82529384, 0.82333773], dtype=np.float32)
BEETLENET_STD = np.array([0.28980458, 0.32252666, 0.3240354], dtype=np.float32)
BEETLENET_AVERAGE_SHAPE = (224, 448)
default_transforms = [Compose([Resize(BEETLENET_AVERAGE_SHAPE), ToTensor(
), Normalize(BEETLENET_MEAN, BEETLENET_STD)])] * 3
batch_size = 32
seed = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
train_transforms = Compose([
    RandomVerticalFlip(p=0.5),
    RandomRotation((-3, 3), fill=255, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    RandomResizedCrop(min_scale=0.95, max_scale=1),
    Resize(BEETLENET_AVERAGE_SHAPE),
    ToTensor(),
    Normalize(BEETLENET_MEAN, BEETLENET_STD)])
modified_transforms = [train_transforms] + default_transforms[1:]


def weighted_pipeline(model_constructor, aux_model_args, taxa, transforms, seed, batch_size,
                      num_workers, device, loss_coefs=None, criterion_constructor=WeightedCCELoss, dataloader_seed=None,
                      optimizer_eps=1e-08, optimizer_lr=0.001,
                      weight_scheme=None, aux_criterion_args={}, pin_memory = True):
    download_dataset()

    generate_taxonomy_dataframe('data/beetles/taxonomy.csv', 'temp_taxonomy-modified.csv', drop_min=9)
    dataset = BeetleSet(csv_path='temp_taxonomy-modified.csv', taxa=taxa)
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
        assert len(loss_coefs) == len(taxa)
    else:
        loss_coefs = torch.ones(len(taxa))

    classes_per_taxon = [len(dataset.labels_dict[taxon]) for taxon in taxa]
    constructor_args = {**aux_model_args, **
                        {'classes_per_taxon': classes_per_taxon}}
    model = model_constructor(**constructor_args).to(device)

    #print(model)
    criterion_args = {**aux_criterion_args, **
                      {'weights_per_taxon': weights_per_taxon}}
    criterion = criterion_constructor(**criterion_args)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=optimizer_lr, eps=optimizer_eps)
    
    return model, optimizer, criterion, dataloaders, taxa, dataset_sizes, loss_coefs


model, optimizer, criterion, dataloaders, taxa, dataset_sizes, loss_coefs = weighted_pipeline(model_constructor, aux_model_args, taxa,
                                modified_transforms, seed, batch_size, num_workers, device,
                                criterion_constructor=WeightedBCELoss, pin_memory=pin_memory)

loss_calculation = weighted_loss(criterion, loss_coefs, reduction)
scaler = torch.cuda.amp.GradScaler(enabled = use_scaler)

def train(data):
    batch_size = len(data[1][0])
    labels = torch.stack(data[1]).to(device) 
    inputs = data[0].to(device)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled = use_amp):
        outputs = model(inputs)
        if not isinstance(outputs, collections.abc.Sequence):
            outputs = [outputs]
        losses = torch.zeros(len(taxa))
        preds = torch.zeros((len(taxa), batch_size),device=device)
        total_loss = 0.0
        for i in range(len(taxa)):
            preds[i] = torch.max(outputs[i], 1)[1]
            loss, reduced_loss = loss_calculation.calc(outputs[i], labels[i], i)
            losses[i] = loss.detach().cpu()
            total_loss += reduced_loss
       
    
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_val)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.empty_cache()
    
    running_losses = losses * inputs.size(0)/dataset_sizes['train']
    running_corrects = torch.sum(preds.detach().cpu() == labels.detach().cpu(), dim=1)
    running_accuracies = running_corrects.double()/dataset_sizes['train']



with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/' + name),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
) as prof:
        for step, batch_data in enumerate(dataloaders['train']):
            if step >= (wait + warmup + active)*repeat:
                break
            train(batch_data)
            prof.step() 
