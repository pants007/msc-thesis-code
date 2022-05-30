import copy
import torch
import torch.nn as nn

class SimpleMultiTaskModel(torch.nn.Module):
    # TODO could consider using dict mapping from species name to number of classes
    def __init__(self, model, classes_per_taxon):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.classes_per_taxon = classes_per_taxon
        num_fc = self.model.fc.in_features
        total_class_num = sum(classes_per_taxon)
        self.model.fc = nn.Linear(num_fc, total_class_num)

    def forward(self, x):
        y = self.model(x)
        outputs = []
        i = 0
        for class_num in self.classes_per_taxon:
            j = i + class_num
            outputs.append(y[:, i: j])
            i = j
        return outputs

class SequentialMultiTaskModel(torch.nn.Module):
    def __init__(self, model, classes_per_taxon):
        assert len(classes_per_taxon) > 0
        super().__init__()
        self.model = copy.deepcopy(model)
        self.fcs = nn.ModuleList()
        in_features = model.fc.in_features
        for class_num in classes_per_taxon:
            self.fcs.append(nn.Linear(in_features, class_num))
            in_features = class_num
        self.model.fc = nn.Identity()

    def forward(self, x):
        output = self.model(x)
        outputs = []
        for fc in self.fcs:
            output =fc(output)
            outputs.append(output)
        return outputs

class SkipMultiTaskModel(torch.nn.Module):
    def __init__(self, model, classes_per_taxon):
        assert len(classes_per_taxon) > 0
        super().__init__()
        self.model = copy.deepcopy(model)
        self.fcs = nn.ModuleList()
        in_features = model.fc.in_features
        for class_num in classes_per_taxon:
            self.fcs.append(nn.Linear(in_features, class_num))
            in_features = class_num + model.fc.in_features
        self.model.fc = nn.Identity()
    
    def forward(self, x):
        x = self.model(x)
        output = torch.empty((x.shape[0], 0), device = x.get_device())
        outputs = []
        for fc in self.fcs:
            output_aug = torch.cat((x, output),dim = 1)
            output = fc(output_aug)
            outputs.append(output)
        return outputs
    
class HiddenLayerModel(torch.nn.Module):
    def __init__(self, model, classes_per_taxon, hidden_features_per_taxon):
        assert len(classes_per_taxon) > 0
        assert len(classes_per_taxon) == len(hidden_features_per_taxon)
        super().__init__()
        self.model = copy.deepcopy(model)
        self.fcs = nn.ModuleList()
        in_features = model.fc.in_features
        for hidden_num, class_num in zip(hidden_features_per_taxon, classes_per_taxon):
            layer = nn.Sequential(
                nn.Linear(in_features, hidden_num),
                nn.ReLU(),
                nn.Linear(hidden_num, class_num))
            self.fcs.append(layer)
            in_features = class_num
        self.model.fc = nn.Identity()
    def forward(self, x):
        output = self.model(x)
        outputs = []
        for fc in self.fcs:
            output = fc(output)
            outputs.append(output)
        return outputs

class HiddenLinearModel(torch.nn.Module):
    def __init__(self, model, classes_per_taxon, hidden_features_per_taxon):
        assert len(classes_per_taxon) > 0
        assert len(classes_per_taxon) == len(hidden_features_per_taxon)
        super().__init__()
        self.model = copy.deepcopy(model)
        self.fcs = nn.ModuleList()
        in_features = model.fc.in_features
        for hidden_num, class_num in zip(hidden_features_per_taxon, classes_per_taxon):
            layer = nn.Sequential(
                nn.Linear(in_features, hidden_num),
                nn.Linear(hidden_num, class_num))
            self.fcs.append(layer)
            in_features = class_num
        self.model.fc = nn.Identity()
    def forward(self, x):
        output = self.model(x)
        outputs = []
        for fc in self.fcs:
            output = fc(output)
            outputs.append(output)
        return outputs