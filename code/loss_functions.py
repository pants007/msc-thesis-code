import torch
import torch.nn.functional as F
import collections.abc


def sanitize_weights(weights_per_taxon):
    if weights_per_taxon is not None:
        assert isinstance(weights_per_taxon, collections.abc.Sequence)
        for taxon_weights in weights_per_taxon:
            assert torch.is_tensor(taxon_weights)
            assert len(taxon_weights.shape) == 1

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, reduction = 'mean', eps = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        pt_dif = torch.clamp(1.0-pt, min = self.eps, max = 1.0 - self.eps)
        focal_loss = pt_dif**self.gamma * ce_loss
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss
class WeightedCCELoss(torch.nn.Module):
    def __init__(self, weights_per_taxon = None):
        super().__init__()
        sanitize_weights(weights_per_taxon)
        self.weights_per_taxon = weights_per_taxon
        self.cce = torch.nn.CrossEntropyLoss(reduction = 'none')
    def forward(self, inputs, targets, t = None):
        cce_loss = self.cce(inputs, targets)
        if self.weights_per_taxon is not None:
            assert t is not None
            target_weights = self.weights_per_taxon[t][targets]
        else:
            target_weights = torch.ones(cce_loss.shape)
        target_weights = target_weights.to(cce_loss.get_device())
        return cce_loss, target_weights
class WeightedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, eps = 1e-8, weights_per_taxon = None):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        sanitize_weights(weights_per_taxon)
        self.weights_per_taxon = weights_per_taxon
        self.cce_module = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets, t = None):
        cce_loss = self.cce_module(inputs, targets)
        pt = torch.exp(-cce_loss)
        pt_dif = torch.clamp(1.0-pt, min = self.eps, max = 1.0 - self.eps)
        focal_loss = pt_dif**self.gamma * cce_loss
        if self.weights_per_taxon is not None:
            assert t is not None
            target_weights = self.weights_per_taxon[t][targets]
        else:
            target_weights = torch.ones(focal_loss.shape)
        target_weights = target_weights.to(cce_loss.get_device())
        return focal_loss, target_weights

class WeightedOneHotLoss(torch.nn.Module):
    def __init__(self, loss_class, weights_per_taxon = None):
        super().__init__()
        sanitize_weights(weights_per_taxon)
        self.weights_per_taxon = weights_per_taxon
        self.loss_module = loss_class(reduction = 'none')
    def forward(self, inputs, targets, t = None):
        one_hot_enc = F.one_hot(targets, inputs.shape[1]).type(inputs.dtype)
        loss = self.loss_module(inputs, one_hot_enc)
        if self.weights_per_taxon is not None:
            assert t is not None
            target_weights = self.weights_per_taxon[t][targets]
            target_weights = target_weights.unsqueeze(dim = 1).repeat(1, inputs.shape[1])
        else:
            target_weights = torch.ones(loss.shape)
        target_weights = target_weights.to(loss.get_device())
        return loss, target_weights
    
class WeightedBCELoss(WeightedOneHotLoss):
    def __init__(self, weights_per_taxon = None):
        super().__init__(torch.nn.BCEWithLogitsLoss, weights_per_taxon)

class WeightedMSELoss(WeightedOneHotLoss):
    def __init__(self, weights_per_taxon = None):
        super().__init__(torch.nn.MSELoss, weights_per_taxon)

class WeightedCenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim, alpha = 0.5, lamda = 0.003, 
                        weight_per_class = None, init_scheme = 'normal'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.lamda = lamda
        if init_scheme == 'normal':
            self.centers = torch.randn(self.num_classes, self.feat_dim)
        else:
            self.centers = torch.rand(self.num_classes, self.feat_dim)
        if weight_per_class is not None:
            assert torch.is_tensor(weight_per_class)
            assert weight_per_class.shape == torch.Size([num_classes])
            
        self.weight_per_class = weight_per_class

        self.mse_module = torch.nn.MSELoss(reduction = 'none')
        
    def forward(self, inputs, targets):
        
        # compute the center loss
        
        target_centers = self.centers[targets]
        loss = self.mse_module(inputs, target_centers) / 2.0
        
        if self.weights_per_taxon is not None:
            target_weights = self.weight_per_class[targets]
            target_weights = target_weights.unsqueeze(1).repeat(1, inputs.shape[1])
        else:
            target_weights = torch.ones(loss.shape)
        
        target_weights = (self.lamda * target_weights).to(loss.get_device())
        
        # update centers
        
        #TODO need to make sure expand does not cause problems later, i.e. make sure there are no in place operations dependencies
        
        #TODO save "classes" matrix when initializing module
        
        #TODO use 'expand' instead of 'repeat' consistently to save time (assuming expand does not cause problems)
        
        #TODO general debugging
        
        classes = torch.arange(self.num_classes).unsqueeze(1)
        classes_expanded = classes.expand(-1, len(targets))
        
        target_hits = classes_expanded == targets
        target_hits_expanded = target_hits.unsqueeze(2).expand(-1, -1, self.feat_dim)
        
        inputs_expanded = inputs.unsqueeze(0).expand((self.class_num,-1, -1))
        
        num = (inputs_expanded * target_hits_expanded).sum(dim = 1)
        denom = 1.0 + target_hits_expanded.sum(dim = 1)
        center_deltas = num/denom
        
        self.centers -= self.alpha * center_deltas
        
        return loss, target_weights
        
        