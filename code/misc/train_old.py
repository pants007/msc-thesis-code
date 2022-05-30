from display import plot_training_stats
from train import weighted_loss
import torch
import numpy as np
import time
import copy
import collections.abc
from pathlib import Path
import shutil
import json

def weighted_fit_terminal(model, criterion, optimizer, scheduler, dataloaders, 
                 taxa, dataset_sizes, device, loss_coefs, num_epochs=25, 
                 plot_path=None, early_stopping=None, reduction = 'mean', 
                 use_amp = True, use_scaler = True, plot_func = None, grad_clip_val = 1e+8):
    temp_folder = '.temp/'
    Path(temp_folder).mkdir(exist_ok = True, parents = True)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    
    metrics = {taxon: { 'train' : {'loss' : [], 'acc' : [] }, 
                        'val'   : {'loss' : [], 'acc' : []}} 
                        for taxon in taxa + ['total']}
    for idx, taxon in enumerate(taxa):
        metrics[taxon]['loss_coef'] = loss_coefs[idx].item()
        
    metrics['epochs'] = []
    metrics['best_epoch'] = 0
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')
    metrics['taxa'] = taxa
    
    render_name =  temp_folder + 'render_temp.png'
    if plot_func is None:
        plot_func = plot_training_stats

    loss_calculation = weighted_loss(criterion, loss_coefs, reduction)
    scaler = torch.cuda.amp.GradScaler(enabled = use_scaler)
    stored_pred = {'train' : np.zeros(( len(taxa), dataset_sizes['train'])),
                        'val' : np.zeros((len(taxa),dataset_sizes['val']))}
    stored_labels = {'train' : np.zeros((len(taxa),dataset_sizes['train'])),
                        'val' : np.zeros((len(taxa),dataset_sizes['val']))}

    store_batch = dataloaders['train'].batch_size
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
                idx = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    batch_size = len(labels[0])
                    labels = torch.stack(labels).to(device) 
                    inputs = inputs.to(device)
                    
                    with torch.cuda.amp.autocast(enabled = use_amp):
                        outputs = model(inputs)
                        if not isinstance(outputs, collections.abc.Sequence):
                            outputs = [outputs]
                        losses = torch.empty(len(taxa))
                        preds = torch.empty((len(taxa), batch_size))
                        total_loss = 0.0
                        weights_sum = 0.0
                        for i in range(len(taxa)):
                            preds[i] = torch.max(outputs[i], 1)[1]
                            loss, reduced_loss = loss_calculation.calc(outputs[i], labels[i], i)
                            losses[i] = loss.detach()
                            total_loss += reduced_loss
                            #total_loss += (taxon_losses * taxon_weights).sum() * loss_coefs[i]
                            #weights_sum += taxon_weights.sum() * loss_coefs[i]
                        #if reduction == 'mean':
                            #total_loss /= weights_sum    
                    # backward + optimize only if in training phase
                    if phase == 'train':        
                        # zero the parameter gradients
                        optimizer.zero_grad(set_to_none = True)
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_val)
                        scaler.step(optimizer)
                        scaler.update()
                    running_losses += losses * inputs.size(0)/dataset_sizes[phase]
                    running_corrects = torch.sum(preds == labels.detach().cpu(), dim=1)
                    running_accuracies += running_corrects.double()/dataset_sizes[phase]

                    stored_pred[phase][:,idx*store_batch:(idx+1)*store_batch] = preds.detach().cpu().numpy()
                    stored_labels[phase][:,idx*store_batch:(idx+1)*store_batch] = labels.detach().cpu().numpy()
                    idx +=1

                total_epoch_loss = running_losses.sum()/len(taxa)
                total_epoch_accuracy = running_accuracies.sum()/len(taxa)
                

                for idx in range(len(taxa)):
                    metrics[taxa[idx]][phase]['loss'].append(running_losses[idx].item())
                    metrics[taxa[idx]][phase]['acc'].append(running_accuracies[idx].item())
                metrics['total'][phase]['loss'].append(total_epoch_loss.item())
                metrics['total'][phase]['acc'].append(total_epoch_accuracy.item())
                    
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

        current_time = time.time() - since
        est_time_left = (current_time/epoch)*num_epochs - current_time
        hours_left = est_time_left//3600
        minutes_left = (est_time_left-(hours_left*3600))//60
        
        #clear_output(wait=True)

        # save to temp folder
        torch.save(best_model_wts, temp_folder + 'state_dict.pt')
        with open(temp_folder + 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent = 4)
        np.save(temp_folder + 'labels_train.npy', stored_labels['train'])
        np.save(temp_folder + 'labels_val.npy', stored_labels['val'])
        np.save(temp_folder + 'preds_train.npy', stored_pred['train'])
        np.save(temp_folder + 'preds_val.npy', stored_pred['val'])

        plot_func(metrics, taxa, [stored_pred, stored_labels], render_name=render_name, path_prefix = temp_folder)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('Estimated time remaining {} hours {} minutes'.format(int(hours_left), int(minutes_left)))
        print('-' * 10)
        print(epoch_printout(metrics, taxa))
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
    plot_func(metrics, taxa, [stored_pred, stored_labels], path_prefix = plot_path,render_name=render_name)
    time_elapsed = time.time() - since

    shutil.rmtree(temp_folder)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('-' * 10)
    print(epoch_printout(metrics, taxa))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics, stored_labels, stored_pred

def weighted_fit(model, criterion, optimizer, scheduler, dataloaders, 
                 taxa, dataset_sizes, device, loss_coefs, num_epochs=25, 
                 plot_path=None, early_stopping=None, reduction = 'mean', 
                 use_amp = True, use_scaler = True, plot_func = None, grad_clip_val = 1e+8):
    temp_folder = '.temp/'
    Path(temp_folder).mkdir(exist_ok = True, parents = True)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    
    metrics = {taxon: { 'train' : {'loss' : [], 'acc' : [] }, 
                        'val'   : {'loss' : [], 'acc' : []}} 
                        for taxon in taxa + ['total']}
    for idx, taxon in enumerate(taxa):
        metrics[taxon]['loss_coef'] = loss_coefs[idx].item()
        
    metrics['epochs'] = []
    metrics['best_epoch'] = 0
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')
    metrics['taxa'] = taxa
    
    render = Rendering()
    render_name =  temp_folder + 'render_temp.png'
    if plot_func is None:
        plot_func = plot_training_stats

    loss_calculation = weighted_loss(criterion, loss_coefs, reduction)
    scaler = torch.cuda.amp.GradScaler(enabled = use_scaler)

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
                    
                    with torch.cuda.amp.autocast(enabled = use_amp):
                        outputs = model(inputs)
                        if not isinstance(outputs, collections.abc.Sequence):
                            outputs = [outputs]
                        losses = torch.empty(len(taxa))
                        preds = torch.empty((len(taxa), batch_size))
                        total_loss = 0.0

                        for i in range(len(taxa)):
                            preds[i] = torch.max(outputs[i], 1)[1]
                            loss, reduced_loss = loss_calculation.calc(outputs[i], labels[i], i)
                            losses[i] = loss.detach()
                            total_loss += reduced_loss
                            #total_loss += (taxon_losses * taxon_weights).sum() * loss_coefs[i]
                            #weights_sum += taxon_weights.sum() * loss_coefs[i]
                        #if reduction == 'mean':
                            #total_loss /= weights_sum    
                    # backward + optimize only if in training phase
                    if phase == 'train':        
                        # zero the parameter gradients
                        optimizer.zero_grad(set_to_none = True)
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_val)
                        scaler.step(optimizer)
                        scaler.update()
                    running_losses += losses * inputs.size(0)/dataset_sizes[phase]
                    running_corrects = torch.sum(preds == labels.detach().cpu(), dim=1)
                    running_accuracies += running_corrects.double()/dataset_sizes[phase]


                total_epoch_loss = running_losses.sum()/len(taxa)
                total_epoch_accuracy = running_accuracies.sum()/len(taxa)
                

                for idx in range(len(taxa)):
                    metrics[taxa[idx]][phase]['loss'].append(running_losses[idx].item())
                    metrics[taxa[idx]][phase]['acc'].append(running_accuracies[idx].item())
                metrics['total'][phase]['loss'].append(total_epoch_loss.item())
                metrics['total'][phase]['acc'].append(total_epoch_accuracy.item())
                    
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

        current_time = time.time() - since
        est_time_left = (current_time/epoch)*num_epochs - current_time
        hours_left = est_time_left//3600
        minutes_left = (est_time_left-(hours_left*3600))//60
        
        #clear_output(wait=True)

        # save to temp folder

        plot_func(metrics, taxa, [], render_name=render_name, path_prefix = temp_folder)
        render.update(filename=render_name)
        render.clear_console()
        render.print('Epoch {}/{}'.format(epoch, num_epochs))
        render.print('Estimated time remaining {} hours {} minutes'.format(int(hours_left), int(minutes_left)))
        render.print('-' * 10)
        render.print(epoch_printout(metrics, taxa))
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
    plot_func(metrics, taxa, [], path_prefix = plot_path,render_name=render_name)
    time_elapsed = time.time() - since

    render.update(filename=render_name)
    shutil.rmtree(temp_folder)
    render.clear_console()
    render.print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    render.print('-' * 10)
    render.print(epoch_printout(metrics, taxa))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics