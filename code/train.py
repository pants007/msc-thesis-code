from IPython.display import clear_output
import torch
import torchvision
from loss_functions import WeightedMSELoss, WeightedBCELoss
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import copy
import collections.abc
from display import epoch_printout

BEETLENET_MEAN = torch.tensor(
    [0.8442649, 0.82529384, 0.82333773])
BEETLENET_STD = torch.tensor([0.28980458, 0.32252666, 0.3240354])

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0, min_epochs: int = 0) -> None:
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epochs = 0
        self.min_epochs = min_epochs

    def __call__(self, val_loss: float) -> None:
        self.epochs += 1
        if self.best_loss == None:
            self.best_loss = val_loss
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            if self.epochs > self.min_epochs:
                self.counter += 1
                print(
                    f"INFO: Early stopping counter {self.counter} of {self.patience}")
                if self.counter >= self.patience:
                    print('INFO: I have no time for your silly games. Stopping early.')
                    self.early_stop = True


class weighted_loss():
    def __init__(self, criterion, loss_coefs, reduction, recon_criterion=None, disc_criterion=None):
        self.criterion = criterion
        #TODO we could just hardcode the WeightedMSELoss and WeightedBCELoss criterions
        if recon_criterion is not None:
            self.recon_criterion = recon_criterion
        if disc_criterion is not None:
            self.disc_criterion = disc_criterion
        weights = False
        if self.criterion.weights_per_taxon is not None:
            weights = True
            self.calc = self.weighted_calc
        else:
            self.calc = self.nonweighted_calc

        # setup loss coefs
        self.loss_coefs = loss_coefs
        coefs = True
        if torch.sum(loss_coefs == torch.ones_like(loss_coefs)) == loss_coefs.size()[0]:
            coefs = False

        # setup reduction type
        self.reduction = True
        if reduction == 'sum':
            self.reduction = False

        if not weights and not coefs and not self.reduction:
            self.calc = self.standard_calc
    
    def denormalize(self, loss_red, losses):
        denorm_loss = loss_red
        if len(losses.shape) > 1:
            denorm_loss = loss_red * np.prod(losses.shape[1:])
        return denorm_loss

    def weighted_calc(self, outputs, labels, i):
        taxon_losses, taxon_weights = self.criterion(outputs, labels, i)
        loss = taxon_losses.mean()
        weighted_losses = taxon_losses * taxon_weights
        reduced_loss = weighted_losses.sum() * self.loss_coefs[i]
        if self.reduction:
            reduced_loss /= taxon_weights.sum() * self.loss_coefs.sum()
        else:
            loss = self.denormalize(loss, taxon_losses)
        return loss, reduced_loss
     
    def nonweighted_calc(self, outputs, labels, i):
        taxon_losses, taxon_weights = self.criterion(outputs, labels, i)
        loss = taxon_losses.mean()
        reduced_loss = taxon_losses.sum() * self.loss_coefs[i]
        if self.reduction:
            reduced_loss /= taxon_weights.sum() * self.loss_coefs.sum()
        else:
            loss = self.denormalize(loss, taxon_losses)
        return loss, reduced_loss

    def standard_calc(self, outputs, labels, i):
        loss, _ = self.criterion(outputs, labels, i)
        return self.denormalize(loss.mean(), loss), loss.sum()

    #TODO we could add a check so that loss coefficients are only used if also coefs = True
    def __extra_loss_calc(self, criterion, outputs, labels, i = None):
        losses = criterion(outputs, labels)
        loss = losses.mean()
        if self.reduction:
            reduced_loss = loss 
            if i is not None:
                reduced_loss *= self.loss_coefs[i] / self.loss_coefs.sum()
        else:
            reduced_loss = losses.sum()
            if i is not None:
                reduced_loss *= self.loss_coefs[i]
            loss = self.denormalize(loss, losses)
        return loss, reduced_loss
    
    def recon_loss_calc(self, outputs, labels, i):
        return self.__extra_loss_calc(self.recon_criterion, outputs, labels, i)

    # TODO we could hardcode the whole adv_loss calculation here
    def adv_loss_calc(self, outputs, labels, i):
        return self.__extra_loss_calc(self.disc_criterion, outputs, labels, i)
    
    # TODO The same for the whole disc loss (consisting of two terms)
    def disc_loss_calc(self, outputs, labels):
        return self.__extra_loss_calc(self.disc_criterion, outputs, labels)
        

def weighted_fit_tb(model, criterion, optimizer, scheduler, dataloaders, 
                 taxa, dataset_sizes, device, loss_coefs, num_epochs=25, 
                 early_stopping=None, reduction = 'mean', 
                 use_amp = True, use_scaler = True, grad_clip_val = 10**4,
                 tb_path='model_stats', tags=[]):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    tb_model_path = tb_path + ''.join(['/' + x for x in tags])

    writer = SummaryWriter(tb_model_path)
    
    model_tag = tags[0] + ''.join(['_' + x for x in tags[1:]])

    metrics = {taxon: { } 
                        for taxon in taxa + ['total']}
    for idx, taxon in enumerate(taxa):
        metrics[taxon]['loss_coef'] = loss_coefs[idx].item()
        
    metrics['best_epoch'] = 0
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')
    metrics['taxa'] = taxa
    
    best_loss = {'train' :np.full((len(taxa)+1), float('inf')) , 'val':np.full((len(taxa)+1), float('inf'))}
    best_acc = {'train' :np.zeros(len(taxa)+1) , 'val':np.zeros(len(taxa)+1)}
    curr_loss = {'train' :np.full((len(taxa)+1), float('inf')) , 'val':np.full((len(taxa)+1), float('inf'))}
    curr_acc = {'train' :np.zeros(len(taxa)+1) , 'val':np.zeros(len(taxa)+1)}
    
    loss_calculation = weighted_loss(criterion, loss_coefs, reduction)
    scaler = torch.cuda.amp.GradScaler(enabled = use_scaler)
    for epoch in range(1, num_epochs + 1):
        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:
            tb_loss = {}
            tb_acc = {}
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
                    batch_size = len(inputs)
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
                            losses[i] = loss.detach().cpu()
                            total_loss += reduced_loss

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
                    curr_loss[phase][idx] = running_losses[idx]
                    curr_acc[phase][idx] = running_accuracies[idx]
                    
                    tb_loss[taxa[idx]] = running_losses[idx]
                    tb_acc[taxa[idx]] = running_accuracies[idx]

                curr_loss[phase][-1] = total_epoch_loss.item()
                curr_acc[phase][-1] = total_epoch_accuracy.item()
                tb_loss['total'] = total_epoch_loss.item()
                tb_acc['total'] = total_epoch_accuracy.item()
                
                writer.add_scalars(model_tag + '/' + phase + '/loss', tb_loss, epoch)
                writer.add_scalars(model_tag + '/' + phase + '/acc', tb_acc, epoch)
                    
                writer.add_scalars('all/' + phase + '/loss', tb_loss, epoch)
                writer.add_scalars('all/' + phase + '/acc', tb_acc, epoch)
                
                best_loss_update = best_loss[phase] > curr_loss[phase]
                best_acc_update = best_acc[phase] < curr_acc[phase]
                
                best_loss[phase][best_loss_update] = curr_loss[phase][best_loss_update]
                best_acc[phase][best_acc_update] = curr_acc[phase][best_acc_update]
                
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

        current_time = time.time() - since
        est_time_left = (current_time/epoch)*num_epochs - current_time
        hours_left = est_time_left//3600
        minutes_left = (est_time_left-(hours_left*3600))//60
        
        
        clear_output(wait=True)

        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('Estimated time remaining {} hours {} minutes'.format(int(hours_left), int(minutes_left)))
        print('-' * 10)
        print(epoch_printout([curr_loss,curr_acc],[best_loss,best_acc], taxa + ['total']))
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
    writer.flush()
    writer.close()
    time_elapsed = time.time() - since

    clear_output(wait=True)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('-' * 10)
    print(epoch_printout([curr_loss,curr_acc],[best_loss,best_acc], taxa + ['total']))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics


def weighted_fit_tb_recon(model, criterion, optimizer, scheduler, dataloaders, 
                            taxa, dataset_sizes, device, loss_coefs, num_epochs=25, 
                            early_stopping=None, reduction = 'mean', 
                            use_amp = True, use_scaler = True, grad_clip_val = 1e+8,
                            tb_path='model_stats', tags=[], disc=None, optim_d=None, optim_d_scheduler = None, 
                            disc_loss_threshold = 0.1, scale_factor = None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    tb_model_path = tb_path + ''.join(['/' + x for x in tags])

    writer = SummaryWriter(tb_model_path)
    #writer.add_graph(model, torch.rand((64,3,224,448), device = device))

    model_tag = tags[0] + ''.join(['_' + x for x in tags[1:]])

    metrics = {taxon: { } 
                        for taxon in taxa + ['total']}
    for idx, taxon in enumerate(taxa):
        metrics[taxon]['loss_coef'] = loss_coefs[idx].item()
        
    metrics['best_epoch'] = 0
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')
    metrics['taxa'] = taxa

    val_display_imgs = None
    val_ground_truth = None
    recon_loss_num_elems = 3 if disc else 1
    extra_loss_coefs = 2 if disc else 1
    
    best_loss = {'train' :np.full((len(taxa)+1 + recon_loss_num_elems), float('inf')), 
                'val':np.full((len(taxa)+1 + recon_loss_num_elems), float('inf'))}
    best_acc = {'train' :np.zeros(len(taxa)+1) , 'val':np.zeros(len(taxa)+1)}
    curr_loss = {'train' :np.full((len(taxa)+1 + recon_loss_num_elems), float('inf')), 
                'val':np.full((len(taxa)+1 + recon_loss_num_elems), float('inf'))}
    curr_acc = {'train' :np.zeros(len(taxa)+1) , 'val':np.zeros(len(taxa)+1)}
    
    #TODO we could just save initialize weighted_loss with WeightedMSELoss and WeightedBCELoss regardless of whether a discriminator is used
    
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction = 'none')
    mse_loss = torch.nn.MSELoss(reduction = 'none')
    if disc:
        loss_calculation = weighted_loss(criterion, loss_coefs, reduction, mse_loss, bce_loss)
    else:
        loss_calculation = weighted_loss(criterion, loss_coefs, reduction, mse_loss)
        
    scaler = torch.cuda.amp.GradScaler(enabled = use_scaler)
    
    input_scaler = None
    if scale_factor is not None:
        input_scaler = torch.nn.Upsample(scale_factor = scale_factor)
    beetle_mean = BEETLENET_MEAN.reshape(-1,1,1).to(device)
    beetle_std = BEETLENET_STD.reshape(-1,1,1).to(device)
    
    
    for epoch in range(1, num_epochs + 1):
        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:
            tb_loss = {}
            tb_acc = {}
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            # forward track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                #TODO we could consider using dictionaries for labels, outputs, loss_coeffs, this would make taxa irrelevant
                running_losses = torch.zeros(len(taxa) + recon_loss_num_elems)
                running_accuracies = torch.zeros(len(taxa))
                first = True
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    batch_size = len(inputs)
                    labels = torch.stack(labels).to(device) 
                    inputs = inputs.to(device)

                    inputs_denorm =  inputs * beetle_std + beetle_mean
                    #inputs_denorm_shifted = (inputs_denorm - 0.5) / 0.5
                    inputs_scaled = inputs_denorm
                    if input_scaler is not None:
                        inputs_scaled = input_scaler(inputs_scaled)
                    
                    with torch.cuda.amp.autocast(enabled = use_amp):
                        outputs = model(inputs)
                        if not isinstance(outputs, collections.abc.Sequence):
                            outputs = [outputs]

                        if phase == 'val' and first:
                            val_display_imgs = outputs[0]
                            val_ground_truth = inputs_scaled
                            first = False

                        losses = torch.empty(len(taxa) + recon_loss_num_elems)
                        preds = torch.empty((len(taxa), batch_size))
                        total_loss = 0.0
                        for i in range(len(taxa)):
                            preds[i] = torch.max(outputs[i + 1], 1)[1]
                            loss, reduced_loss = loss_calculation.calc(outputs[i + 1], labels[i], i)
                            losses[i] = loss.detach().cpu()
                            total_loss += reduced_loss
                        if disc:
                            disc_real = disc(inputs_scaled)
                            # Detach to stop autograd at this value
                            disc_fake = disc(outputs[0].detach())
                            
                            #TODO the loss calculation for loss_adv and loss_disc could be incorporated further into the 'weighted_loss' class
                            ones = torch.ones(disc_fake.shape, device=device)
                            zeros = torch.zeros(disc_fake.shape, device=device)

                            loss_adv, loss_adv_reduced = loss_calculation.adv_loss_calc(disc_fake,ones, -1)
                            loss_disc_real, loss_disc_real_reduced = loss_calculation.disc_loss_calc(disc_real, ones)
                            loss_disc_fake, loss_disc_fake_reduced = loss_calculation.disc_loss_calc(disc_fake, zeros)
                            loss_disc = (loss_disc_real + loss_disc_fake) / 2
                            denom = 2 if reduction == 'mean' else 1
                            loss_disc_reduced = (loss_disc_real_reduced + loss_disc_fake_reduced) / denom
                    
                    # backward + optimize only if in training phase
                    if disc and phase == 'train' and loss_disc_reduced / loss_adv_reduced > disc_loss_threshold:
                    # use the mixed precision scaler for the actual update
                        optim_d.zero_grad(set_to_none = True)
                        scaler.scale(loss_disc_reduced).backward()
                        scaler.unscale_(optim_d)
                        torch.nn.utils.clip_grad_value_(disc.parameters(), grad_clip_val)
                        scaler.step(optim_d)
                        scaler.update()


                    # Calculate autograd and loss with mixed precision
                    with torch.cuda.amp.autocast(enabled = use_amp):
                        loss_adv_reduced = 0.0
                        if disc:
                            disc_fake = disc(outputs[0])
                            loss_adv, loss_adv_reduced = loss_calculation.adv_loss_calc(disc_fake, ones, -1)
                        recon_loss, recon_loss_reduced = loss_calculation.recon_loss_calc(outputs[0], inputs_scaled, -extra_loss_coefs)
                        total_loss += recon_loss_reduced + loss_adv_reduced

                    if phase == 'train':       
                        # zero the parameter gradients
                        optimizer.zero_grad(set_to_none = True)
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_val)
                        scaler.step(optimizer)
                        scaler.update()
                    extra_losses = recon_loss
                    if disc:
                        extra_losses = torch.tensor([recon_loss, loss_adv, loss_disc])
                    losses[-recon_loss_num_elems:] = extra_losses
                    running_losses += losses * inputs.size(0)/dataset_sizes[phase]
                    running_corrects = torch.sum(preds == labels.detach().cpu(), dim=1)
                    running_accuracies += running_corrects.double()/dataset_sizes[phase]

                total_epoch_loss = running_losses.sum()/(len(taxa) + recon_loss_num_elems)
                total_epoch_accuracy = running_accuracies.sum()/len(taxa)
                for idx in range(len(taxa)):
                    curr_loss[phase][idx] = running_losses[idx]
                    curr_acc[phase][idx] = running_accuracies[idx]
                    
                    tb_loss[taxa[idx]] = running_losses[idx]
                    tb_acc[taxa[idx]] = running_accuracies[idx]

                curr_loss[phase][-1-recon_loss_num_elems:-1] = running_losses[-recon_loss_num_elems:].detach().numpy()

                tb_loss['recon_loss'] = running_losses[-recon_loss_num_elems]
                if disc:
                    tb_loss['adv_loss'] = running_losses[-2]
                    tb_loss['disc_loss'] = running_losses[-1]
                
                curr_loss[phase][-1] = total_epoch_loss.item()
                curr_acc[phase][-1] = total_epoch_accuracy.item()
                tb_loss['total'] = total_epoch_loss.item()
                tb_acc['total'] = total_epoch_accuracy.item()
                
                writer.add_scalars(model_tag + '/' + phase + '/loss', tb_loss, epoch)
                writer.add_scalars(model_tag + '/' + phase + '/acc', tb_acc, epoch)
                    
                writer.add_scalars('all/' + phase + '/loss', tb_loss, epoch)
                writer.add_scalars('all/' + phase + '/acc', tb_acc, epoch)
                
                best_loss_update = best_loss[phase] > curr_loss[phase]
                best_acc_update = best_acc[phase] < curr_acc[phase]
                
                best_loss[phase][best_loss_update] = curr_loss[phase][best_loss_update]
                best_acc[phase][best_acc_update] = curr_acc[phase][best_acc_update]

                if phase == 'train':
                    scheduler.step()
                    if optim_d_scheduler is not None:
                        optim_d_scheduler.step()
                    
                else:
                    if total_epoch_loss < metrics['best_loss']:
                        metrics['best_loss'] = total_epoch_loss.item()
                        metrics['best_acc'] = total_epoch_accuracy.item()
                        metrics['best_epoch'] = epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if early_stopping is not None:
                        early_stopping(total_epoch_loss)
        
        #val_display_imgs = val_display_imgs * 0.5 + 0.5

        display_imgs_gen = torchvision.utils.make_grid(val_display_imgs, nrow=8)
        display_imgs_truth = torchvision.utils.make_grid(val_ground_truth, nrow=8)
        imgs_height = int(display_imgs_gen.shape[1]/(dataloaders['val'].batch_size//8))
        tb_imgs = torch.zeros((3, imgs_height*4,display_imgs_gen.shape[2]))
        tb_imgs[:,0:imgs_height] = display_imgs_gen[:,0:imgs_height,:]
        tb_imgs[:,imgs_height:imgs_height*2] = display_imgs_truth[:,0:imgs_height,:]
        tb_imgs[:,imgs_height*2:imgs_height*3] = display_imgs_gen[:,imgs_height:imgs_height*2] 
        tb_imgs[:,imgs_height*3:imgs_height*4] = display_imgs_truth[:,imgs_height:imgs_height*2] 
        writer.add_image(model_tag + '/val_recon_imgs', tb_imgs, epoch)

        current_time = time.time() - since
        est_time_left = (current_time/epoch)*num_epochs - current_time
        hours_left = est_time_left//3600
        minutes_left = (est_time_left-(hours_left*3600))//60
        
        
        clear_output(wait=True)

        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('Estimated time remaining {} hours {} minutes'.format(int(hours_left), int(minutes_left)))
        print('-' * 10)
        #TODO we should call rename epoch_printout to epoch_printout_recons and then restore the old recon so we can use it with old training
        print(epoch_printout([curr_loss,curr_acc],[best_loss,best_acc], taxa, recon_loss_num_elems))
        #print('Epoch {}/{}'.format(epoch, num_epochs))
        #print('-' * 10)
        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #'Train', total_train_loss, total_train_acc))
        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #'Val', total_val_loss, total_val_acc))
        #print()

        if early_stopping is not None and early_stopping.early_stop:
            break
    
    clear_output(wait=True)
    writer.flush()
    writer.close()
    time_elapsed = time.time() - since

    clear_output(wait=True)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('-' * 10)
    print(epoch_printout([curr_loss,curr_acc],[best_loss,best_acc], taxa, recon_loss_num_elems))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics

def weighted_test(model, criterion, test_loader, device, dataset_sizes, taxa):
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
                taxon_losses, _ = criterion(outputs[i], labels[i], i)
                losses[i] = taxon_losses.mean()
            running_losses += losses * inputs.size(0)/dataset_sizes['test']
            running_corrects = torch.sum(preds == labels.cpu(), dim=1)
            running_accuracies += running_corrects.double()/dataset_sizes['test']               
        total_loss = running_losses.sum()/len(taxa)
        total_accuracy = running_accuracies.sum()/len(taxa)
    test_dict = {taxon : {'loss' : running_losses[idx].item(), 
                          'acc' : running_accuracies[idx].item()} 
                    for idx, taxon in enumerate(taxa)}
    test_dict['total'] = {'loss' : total_loss.item(), 'acc' : total_accuracy.item()}

    return test_dict


def weighted_test_recon(model, criterion, test_loader, device, dataset_sizes, taxa,
                        tb_path, tags, disc = None, scale_factor = None, imgs_per_row = 8, reduction = 'mean'):
    model.eval()

    tb_model_path = tb_path + ''.join(['/' + x for x in tags])
    writer = SummaryWriter(tb_model_path)

    model_tag = tags[0] + ''.join(['_' + x for x in tags[1:]])

    with torch.no_grad():
        extra_loss_elems = 3 if disc else 1
        extra_loss_coefs = 2 if disc else 1
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        mse_loss = torch.nn.MSELoss(reduction='none')
        if disc:
            loss_calculation = weighted_loss(criterion, torch.ones(
                len(taxa) + extra_loss_coefs), reduction, mse_loss, bce_loss)
        else:
            loss_calculation = weighted_loss(criterion, torch.ones(len(taxa) + extra_loss_coefs), reduction, mse_loss)
        input_scaler = None
        if scale_factor is not None:
            input_scaler = torch.nn.Upsample(scale_factor = scale_factor)
        
        running_losses = torch.zeros(len(taxa) + extra_loss_elems)
        running_accuracies = torch.zeros(len(taxa))
        first = True
        for inputs, labels in test_loader:
            batch_size = len(labels[0])
            labels = torch.stack(labels).to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            if not isinstance(outputs, collections.abc.Sequence):
                outputs = [outputs]
            
            _,_,h,w = inputs.shape
            beetlenet_std_exp = BEETLENET_STD[:, None, None].expand(3, h, w).to(device)
            beetlenet_mean_exp = BEETLENET_MEAN[:, None, None].expand(3, h, w).to(device)
            inputs_denorm =  inputs * beetlenet_std_exp + beetlenet_mean_exp
            inputs_scaled = inputs_denorm
            if input_scaler is not None:
                inputs_scaled = input_scaler(inputs_denorm)
            if first:
                imgs = outputs[0]
                ground_truth = inputs_scaled
                first = False
            losses = torch.empty((len(taxa)) + extra_loss_elems)
            preds = torch.empty((len(taxa), batch_size))
            for i in range(len(taxa)):
                preds[i] = torch.max(outputs[i + 1], 1)[1]
                loss = loss_calculation.calc(outputs[i + 1], labels[i], i)[0]
                losses[i] = loss 
            
            losses[-extra_loss_elems] = loss_calculation.recon_loss_calc(outputs[0], inputs_scaled, -extra_loss_coefs)[0]
            if disc is not None:
                disc_fake = disc(outputs[0])
                disc_real = disc(inputs_scaled)
                ones = torch.ones(disc_fake.shape, device=device)
                zeros = torch.zeros(disc_fake.shape, device=device)
                adv_loss = loss_calculation.adv_loss_calc(disc_fake, ones, -1)[0]
                disc_loss = (loss_calculation.disc_loss_calc(disc_real, ones)[0] + loss_calculation.disc_loss_calc(disc_fake, zeros)[0]) /2
                losses[-1] = disc_loss
                losses[-2] = adv_loss
            running_losses += losses * inputs.size(0)/dataset_sizes['test']
            running_corrects = torch.sum(preds == labels.cpu(), dim=1)
            running_accuracies += running_corrects.double() / \
                dataset_sizes['test']

        display_imgs_gen = torchvision.utils.make_grid(imgs, nrow=imgs_per_row)
        display_imgs_truth = torchvision.utils.make_grid(ground_truth, nrow=imgs_per_row)
        imgs_height = int(display_imgs_gen.shape[1]/(test_loader.batch_size//imgs_per_row))
        tb_imgs = torch.zeros((3, imgs_height*4,display_imgs_gen.shape[2]))
        tb_imgs[:,0:imgs_height] = display_imgs_gen[:,0:imgs_height,:]
        tb_imgs[:,imgs_height:imgs_height*2] = display_imgs_truth[:,0:imgs_height,:]
        tb_imgs[:,imgs_height*2:imgs_height*3] = display_imgs_gen[:,imgs_height:imgs_height*2] 
        tb_imgs[:,imgs_height*3:imgs_height*4] = display_imgs_truth[:,imgs_height:imgs_height*2]
        writer.add_image(model_tag + '/test_recon_imgs', tb_imgs, 0)

        total_loss = running_losses.sum()/(len(taxa) + extra_loss_elems)
        total_accuracy = running_accuracies.sum()/len(taxa)
    test_dict = {taxon: {'loss': running_losses[idx].item(),
                         'acc': running_accuracies[idx].item()}
                 for idx, taxon in enumerate(taxa)}
    test_dict['recon_loss'] = running_losses[-extra_loss_elems].item()
    test_dict['adv_loss'] = None
    test_dict['disc_loss'] = None
    if disc is not None:
        test_dict['disc_loss'] = running_losses[-1].item()
        test_dict['adv_loss'] = running_losses[-2].item()

    test_dict['total'] = {
        'loss': total_loss.item(), 'acc': total_accuracy.item()}

    return test_dict
