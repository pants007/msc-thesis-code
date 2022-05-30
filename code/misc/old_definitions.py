import torch.nn
import torch
from ..dataset import download_dataset, generate_taxonomy_dataframe, BeetleSet, get_dataloaders
from pathlib import Path
import time
import copy
import json
import collections.abc
import torch.nn.functional as F

def pipeline(model_constructor, constructor_args, model_name, taxa, transforms, seed, batch_size,
             num_workers, device, num_epochs, loss_coefs=None, root='../output/msc-thesis-22/models/',
             criterion=torch.nn.CrossEntropyLoss(), early_stopping=None, one_hot=False,
             loss_sum=False, dataloader_seed=None, optimizer_eps=1e-08, optimizer_lr=0.001,
             scheduler_gamma=0.995):
    download_dataset()
    tmap = {'subfamily': '0', 'tribe': '1', 'genus': '2', 'species': '3'}
    model_name_ext = model_name + '_' + \
        ''.join([tmap[taxon] for taxon in taxa])
    Path(root + model_name_ext).mkdir(parents=True, exist_ok=True)
    generate_taxonomy_dataframe('data/beetles/taxonomy.csv',
                                root + model_name_ext + '/taxonomy-modified.csv', drop_min=9)
    dataset = BeetleSet(csv_path=root + model_name_ext +
                        '/taxonomy-modified.csv', taxa=taxa)
    dataloaders, dataset_sizes = get_dataloaders(dataset, 0.2, transforms, None, batch_size,
                                                 num_workers, seed, dataloader_seed)
    classes_per_taxon = [len(dataset.labels_dict[taxon]) for taxon in taxa]
    constructor_args = {**constructor_args, **
                        {'classes_per_taxon': classes_per_taxon}}
    model = model_constructor(**constructor_args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=optimizer_lr, eps=optimizer_eps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=scheduler_gamma)
    if __name__ == '__main__':
        model, metrics = fit(model, criterion, optimizer, scheduler, dataloaders, taxa, dataset_sizes,
                             device, loss_coefs, num_epochs, plot_path=root + model_name_ext + '/metrics',
                             early_stopping=early_stopping, one_hot=one_hot, loss_sum=loss_sum)
    test_metrics = test(model, criterion, dataloaders['test'], device,
                        dataset_sizes, taxa, one_hot=one_hot, loss_sum=loss_sum)
    for taxon in taxa + ['total']:
        metrics[taxon]['test'] = test_metrics[taxon]
    torch.save(model.state_dict(), root + model_name_ext + '/state_dict.pt')
    with open(root + model_name_ext + '/metrics.json', 'w') as file:
        json.dump(metrics, file, indent=4)
    return model, metrics


def fit(model, criterion, optimizer, scheduler, dataloaders, taxa, dataset_sizes,
        device, loss_coefs=None, num_epochs=25, plot_path=None, early_stopping=None,
        one_hot=False, loss_sum=False):
    if loss_coefs is not None:
        assert len(loss_coefs) == len(taxa)
    else:
        loss_coefs = torch.ones(len(taxa))
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    metrics = {taxon: {'train': {'loss': [], 'acc': []},
                       'val': {'loss': [], 'acc': []}}
               for taxon in taxa + ['total']}
    for idx, taxon in enumerate(taxa):
        metrics[taxon]['loss_coef'] = loss_coefs[idx].item()
    metrics['epochs'] = []
    metrics['best_epoch'] = 0
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')

    render = Rendering()
    render_name = 'render_temp.png'

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, num_epochs + 1):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            # forward track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                #TODO we could consider using dictionaries for labels, outputs, loss_coeffs, this would make taxa irrelevant
                running_losses = torch.zeros(len(taxa))
                running_accuracies = torch.zeros(len(taxa))

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    batch_size = len(labels[0])
                    labels = torch.stack(labels).to(device)
                    inputs = inputs.to(device)

                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        if not isinstance(outputs, collections.abc.Sequence):
                            outputs = [outputs]
                        losses = torch.empty(len(taxa))
                        preds = torch.empty((len(taxa), batch_size))

                        for i in range(len(taxa)):
                            preds[i] = torch.max(outputs[i], 1)[1]
                            if one_hot:
                                vecs = F.one_hot(labels[i], outputs[i].shape[1]).type(
                                    outputs[i].dtype)
                                losses[i] = criterion(outputs[i], vecs)
                            else:
                                losses[i] = criterion(outputs[i], labels[i])
                        if loss_sum:
                            total_loss = torch.sum(
                                loss_coefs * losses).to(device)

                        else:
                            total_loss = (
                                torch.sum(loss_coefs * losses)/len(losses)).to(device)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    if loss_sum:
                        if one_hot:
                            for i in range(len(taxa)):
                                losses[i] /= outputs[i].shape[1]
                        running_losses += losses/dataset_sizes[phase]
                    else:
                        running_losses += losses * \
                            inputs.size(0)/dataset_sizes[phase]
                    running_corrects = torch.sum(
                        preds == labels.detach().cpu(), dim=1)
                    running_accuracies += running_corrects.double() / \
                        dataset_sizes[phase]
                total_epoch_loss = running_losses.sum()/len(taxa)
                total_epoch_accuracy = running_accuracies.sum()/len(taxa)

                for idx in range(len(taxa)):
                    metrics[taxa[idx]][phase]['loss'].append(
                        running_losses[idx].item())
                    metrics[taxa[idx]][phase]['acc'].append(
                        running_accuracies[idx].item())
                metrics['total'][phase]['loss'].append(total_epoch_loss.item())
                metrics['total'][phase]['acc'].append(
                    total_epoch_accuracy.item())

                if phase == 'train':
                    scheduler.step()

                else:
                    if total_epoch_loss < metrics['best_loss']:
                        metrics['best_loss'] = total_epoch_loss.item()
                        metrics['best_acc'] = total_epoch_accuracy.item()
                        metrics['best_epoch'] = epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if early_stopping is not None:
                        early_stopping(total_epoch_loss)

        metrics['epochs'].append(epoch)

        total_train_loss = metrics['total']['train']['loss'][-1]
        total_train_acc = metrics['total']['train']['acc'][-1]
        total_val_loss = metrics['total']['val']['loss'][-1]
        total_val_acc = metrics['total']['val']['acc'][-1]

        #clear_output(wait=True)
        plot_training_stats(metrics, taxa, render_name=render_name)
        render.update(filename=render_name)
        render.clear_console()
        render.print('Epoch {}/{}'.format(epoch, num_epochs))
        render.print('-' * 10)
        render.print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Train', total_train_loss, total_train_acc))
        render.print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Val', total_val_loss, total_val_acc))

        #print('Epoch {}/{}'.format(epoch, num_epochs))
        #print('-' * 10)
        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #'Train', total_train_loss, total_train_acc))
        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #'Val', total_val_loss, total_val_acc))
        #print()

        if early_stopping is not None and early_stopping.early_stop:
            break

    #clear_output(wait=True)
    plot_training_stats(metrics, taxa, path_prefix=plot_path,
                        render_name=render_name)
    time_elapsed = time.time() - since

    render.update(filename=render_name)
    render.clear_console()
    render.print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    render.print('-' * 10)
    render.print('Best validation epoch: {:4d}'.format(metrics['best_epoch']))
    render.print('Best validation accuracy: {:4f}'.format(metrics['best_acc']))
    render.print('Best validation loss: {:4f}'.format(metrics['best_loss']))

    #print('Training complete in {:.0f}m {:.0f}s'.format(
    #time_elapsed // 60, time_elapsed % 60))
    #print('-' * 10)
    #print('Best validation epoch: {:4d}'.format(metrics['best_epoch']))
    #print('Best validation accuracy: {:4f}'.format(metrics['best_acc']))
    #print('Best validation loss: {:4f}'.format(metrics['best_loss']))

    #print()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics

def weighted_pipeline(model_constructor, aux_model_args, model_name, taxa, transforms, seed, batch_size,
                      num_workers, device, num_epochs, loss_coefs=None, root='../output/msc-thesis-22/models/', criterion_constructor=WeightedCCELoss, early_stopping=None, dataloader_seed=None, 
                      optimizer_constructor = torch.optim.Adam,  aux_optimizer_args = {}, 
                      scheduler_constructor = torch.optim.lr_scheduler.ExponentialLR, 
                      aux_scheduler_args = {'gamma': 0.995}, reduction='mean', weight_scheme=None, 
                      aux_criterion_args={}, use_amp = True, use_scaler=True, pin_memory = True, 
                      plot_func = None, grad_clip_val = 1e+8, train=True):
    download_dataset()

    tmap = {'subfamily': '0', 'tribe': '1', 'genus': '2', 'species': '3'}
    model_name_ext = model_name + '_' + \
        ''.join([tmap[taxon] for taxon in taxa])
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
    
    optimizer_args = {**aux_optimizer_args, **
                      {'params': model.parameters()}}
    optimizer = optimizer_constructor(**optimizer_args)
    
    scheduler_args = {**aux_scheduler_args, **
                    {'optimizer': optimizer}}
    
    scheduler = scheduler_constructor(**scheduler_args)

    if __name__ == '__main__' and train:
            model, metrics, stored_labels, stored_preds = weighted_fit(model, criterion, optimizer, scheduler, 
                                                                        dataloaders, taxa, dataset_sizes,
                                        device, loss_coefs, num_epochs, plot_path=root + model_name_ext + '/metrics',
                                        early_stopping=early_stopping, reduction=reduction, use_amp=use_amp,
                                            use_scaler=use_scaler, plot_func=plot_func, grad_clip_val = grad_clip_val)
    else:
        return model, {}
    
    torch.save(model.state_dict(), root + model_name_ext + '/state_dict.pt')
    np.save(root + model_name_ext + '/labels_train', stored_labels['train'])
    np.save(root + model_name_ext + '/labels_val', stored_labels['val'])
    np.save(root + model_name_ext + '/preds_train', stored_preds['train'])
    np.save(root + model_name_ext + '/preds_val', stored_preds['val'])
    criterion.weights_per_taxon = None
    test_metrics = weighted_test(model, criterion, dataloaders['test'], device,
                                 dataset_sizes, taxa)
    for taxon in taxa + ['total']:
        metrics[taxon]['test'] = test_metrics[taxon]

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

    metrics['batch_size'] = batch_size

    metrics['seed'] = seed

    with open(root + model_name_ext + '/metrics.json', 'w') as file:
        json.dump(metrics, file, indent=4)

    return model, metrics


def weighted_pipeline_tb(model_constructor, aux_model_args, model_name, taxa, transforms, seed, batch_size,
                      num_workers, device, num_epochs, tags, loss_coefs=None, root='../output/msc-thesis-22/models/', criterion_constructor=WeightedCCELoss, early_stopping=None, dataloader_seed=None, 
                      optimizer_constructor = torch.optim.Adam,  aux_optimizer_args = {}, 
                      scheduler_constructor = torch.optim.lr_scheduler.ExponentialLR, 
                      aux_scheduler_args = {'gamma': 0.995}, reduction='mean', weight_scheme=None, 
                      aux_criterion_args={}, use_amp = True, use_scaler=True, pin_memory = True, 
                      grad_clip_val = 1e+8, train=True, tb_root = '../output/msc-thesis-22/tensorboard/'):
    download_dataset()

    tmap = {'subfamily': '0', 'tribe': '1', 'genus': '2', 'species': '3'}
    model_name_ext = model_name + '_' + \
        ''.join([tmap[taxon] for taxon in taxa])
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
    
    optimizer_args = {**aux_optimizer_args, **
                      {'params': model.parameters()}}
    optimizer = optimizer_constructor(**optimizer_args)
    
    scheduler_args = {**aux_scheduler_args, **
                    {'optimizer': optimizer}}
    
    scheduler = scheduler_constructor(**scheduler_args)

    if __name__ == '__main__' and train:
            model, metrics = weighted_fit_tb(model, criterion, optimizer, scheduler, dataloaders, taxa, dataset_sizes,
                                            device, loss_coefs, num_epochs, early_stopping=early_stopping, reduction=reduction, 
                                            use_amp=use_amp, use_scaler=use_scaler, grad_clip_val = grad_clip_val, tb_path=tb_root, 
                                            tags=tags)
    else:
        return model, {}
    
    torch.save(model.state_dict(), root + model_name_ext + '/state_dict.pt')
    criterion.weights_per_taxon = None
    test_metrics = weighted_test(model, criterion, dataloaders['test'], device,
                                 dataset_sizes, taxa)
    for taxon in taxa + ['total']:
        metrics[taxon]['test'] = test_metrics[taxon]

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

    metrics['batch_size'] = batch_size

    metrics['seed'] = seed

    with open(root + model_name_ext + '/metrics.json', 'w') as file:
        json.dump(metrics, file, indent=4)

    return model, metrics

def test(model, criterion, test_loader, device, dataset_sizes,
         taxa, one_hot=False, loss_sum=False):
    model.eval()
    with torch.no_grad():
        running_losses = torch.zeros(len(taxa))
        running_accuracies = torch.zeros(len(taxa))
        for inputs, labels in test_loader:
            batch_size = len(labels[0])
            labels = torch.stack(labels).to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            if not isinstance(outputs, collections.abc.Sequence):
                outputs = [outputs]

            losses = torch.empty((len(taxa)))
            preds = torch.empty((len(taxa), batch_size))
            for i in range(len(taxa)):
                preds[i] = torch.max(outputs[i], 1)[1]
                if one_hot:
                    vecs = F.one_hot(labels[i], outputs[i].shape[1]).type(
                        outputs[i].dtype)
                    losses[i] = criterion(outputs[i], vecs)
                else:
                    losses[i] = criterion(outputs[i], labels[i])
            if loss_sum:
                running_losses += losses / dataset_sizes['test']
            else:
                running_losses += losses * inputs.size(0)/dataset_sizes['test']
            running_corrects = torch.sum(preds == labels.cpu(), dim=1)
            running_accuracies += running_corrects.double() / \
                dataset_sizes['test']
        total_loss = running_losses.sum()/len(taxa)
        total_accuracy = running_accuracies.sum()/len(taxa)
    test_dict = {taxon: {'loss': running_losses[idx].item(),
                         'acc': running_accuracies[idx].item()}
                 for idx, taxon in enumerate(taxa)}
    test_dict['total'] = {
        'loss': total_loss.item(), 'acc': total_accuracy.item()}

    return test_dict
