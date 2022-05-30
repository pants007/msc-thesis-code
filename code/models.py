import copy
import torch
import torch.nn as nn
import torchvision


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
    # TODO could consider using dict mapping from species name to number of classes
    def __init__(self, model, classes_per_taxon):
        assert len(classes_per_taxon) > 0
        super().__init__()
        self.model = copy.deepcopy(model)
        out_features = classes_per_taxon[0]
        self.model.fc = nn.Linear(model.fc.in_features, out_features)
        self.aux_fcs = nn.ModuleList()
        in_features = out_features
        for class_num in classes_per_taxon[1:]:
            self.aux_fcs.append(nn.Linear(in_features, class_num))
            in_features = class_num
        
    def forward(self, x):
        output = self.model(x)
        outputs = [output]
        for aux_fc in self.aux_fcs:
            output = aux_fc(output)
            outputs.append(output)
        return outputs

class SkipMultiTaskModel(torch.nn.Module):
    def __init__(self, model, classes_per_taxon):
        assert len(classes_per_taxon) > 0
        super().__init__()
        self.model = copy.deepcopy(model)
        self.aux_fcs = nn.ModuleList()
        in_features = model.fc.in_features
        for class_num in classes_per_taxon:
            self.aux_fcs.append(nn.Linear(in_features, class_num))
            in_features = class_num + model.fc.in_features
        self.model.fc = nn.Identity()
    
    def forward(self, x):
        x = self.model(x)
        output = torch.empty((x.shape[0], 0), device = x.get_device())
        outputs = []
        for aux_fc in self.aux_fcs:
            output_aug = torch.cat((x, output),dim = 1)
            output = aux_fc(output_aug)
            outputs.append(output)
        return outputs
    
class HiddenLayerModel(torch.nn.Module):
    def __init__(self, model, classes_per_taxon, hidden_features_per_taxon):
        assert len(classes_per_taxon) > 0
        assert len(classes_per_taxon) == len(hidden_features_per_taxon)
        super().__init__()
        self.model = copy.deepcopy(model)
        hidden_features = hidden_features_per_taxon[0]
        out_features = classes_per_taxon[0]
        self.model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )
        self.aux_fcs = nn.ModuleList()
        in_features = out_features
        for hidden_num, class_num in zip(hidden_features_per_taxon[1:], classes_per_taxon[1:]):
            layer = nn.Sequential(
                nn.Linear(in_features, hidden_num),
                nn.ReLU(),
                nn.Linear(hidden_num, class_num)
            )
            self.aux_fcs.append(layer)
            in_features = class_num
    def forward(self, x):
        output = self.model(x)
        outputs = [output]
        for aux_fc in self.aux_fcs:
            output = aux_fc(output)
            outputs.append(output)
        return outputs

class HiddenLinearModel(torch.nn.Module):
    def __init__(self, model, classes_per_taxon, hidden_features_per_taxon):
        assert len(classes_per_taxon) > 0
        assert len(classes_per_taxon) == len(hidden_features_per_taxon)
        super().__init__()
        self.model = copy.deepcopy(model)
        hidden_features = hidden_features_per_taxon[0]
        out_features = classes_per_taxon[0]
        self.model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_features),
            nn.Linear(hidden_features, out_features)
        )
        self.aux_fcs = nn.ModuleList()
        in_features = out_features
        for hidden_num, class_num in zip(hidden_features_per_taxon[1:], classes_per_taxon[1:]):
            layer = nn.Sequential(
                nn.Linear(in_features, hidden_num),
                nn.Linear(hidden_num, class_num)
            )
            self.aux_fcs.append(layer)
            in_features = class_num
    def forward(self, x):
        output = self.model(x)
        outputs = [output]
        for aux_fc in self.aux_fcs:
            output = aux_fc(output)
            outputs.append(output)
        return outputs


class SequentialHierarchicalModel(torch.nn.Module):
    def __init__(self, model, classes_per_taxon):
        assert len(classes_per_taxon) > 0
        super().__init__()
        self.stem = nn.Sequential(
            copy.deepcopy(model.conv1), 
            copy.deepcopy(model.bn1), 
            copy.deepcopy(model.relu), 
            copy.deepcopy(model.maxpool))
        self.blocks = nn.ModuleList()
        block_list = [model.layer1, model.layer2, model.layer3, model.layer4]
        self.class_fcs = nn.ModuleList()
        self.combi_fcs = nn.ModuleList()
        fc_features = 64
        for i, num_classes in enumerate(classes_per_taxon):
            self.blocks.append(copy.deepcopy(block_list[i]))
            self.class_fcs.append(
                nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), 
                              nn.Flatten(), 
                              nn.Linear(fc_features, num_classes)))
            self.combi_fcs.append(nn.Linear(num_classes,fc_features))
            fc_features *= 2

        
    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for i, block in enumerate(self.blocks):
            #combine output from previous block with feature maps from previous block
            x = block(x)
            res_connection = x
            x = self.class_fcs[i](x)
            outputs.append(x)
            x = self.combi_fcs[i](x).view(-1, self.combi_fcs[i].out_features,1,1)
            x = res_connection * x #potentially add activation function
        return outputs

class FusionModel(torch.nn.Module):
    def __init__(self, classes_per_taxon, graph_dict):
        super().__init__()
        self.classes_per_taxon = classes_per_taxon
        ref_dict = copy.deepcopy(graph_dict)
        number_channels = 16
        for layer in ref_dict:
            number_track = len(ref_dict[layer])
            for block in ref_dict[layer]:
                if block != 'name':
                    ref_dict[layer][block]['class'] = self.instantiate_class(ref_dict[layer][block], number_channels, number_track)
            number_channels *= 2
        #print(ref_dict)
        self.modulelist = nn.ModuleList()
        for layer in ref_dict:
            layerlist = nn.ModuleList()
            for block in ref_dict[layer]:
                if block != 'name': 
                    child_list = []
                    for child in ref_dict[layer][block]['children']:
                        keys = child.split(':')
                        child_list.append(ref_dict[keys[0]][keys[1]]['class'])
                    ref_dict[layer][block]['class'].edges = child_list
                    layerlist.append(ref_dict[layer][block]['class'])
            self.modulelist.append(layerlist)
        #self.module_dict = nn.ModuleDict(ref_dict)
    
    def instantiate_class(self, block, number_channel, number_track):
        if block['class'] == 'FusionRoot':
            return FusionRoot()
        if block['class'] == 'FusionRootWithStem':
            return FusionRootWithStem()
        if block['class'] == 'FusionFirstLayer':
            return FusionFirstLayer(block['blocks'])
        if block['class'] == 'FusionLayer':
            return FusionLayer(number_channel, number_channel*2, block['blocks'])
        if block['class'] == 'FusionStem':
            return FusionStem(block['blocks'])
        if block['class'] == 'FusionStemWithArgs':
            return FusionStem(block['blocks'], stride=block['stride'], out_channels=block['out_channels'])
        if block['class'] == 'FusionDownsample':
            return FusionDownsample(block['input_channels'], 2*number_channel, block['stride'])
        if block['class'] == 'FusionDownsampleAvgPool':
            return FusionDownsampleAvgPool(stride=block['stride'])
        if block['class'] == 'FusionDownsampleMaxPool':
            return FusionDownsampleMaxPool(stride=block['stride'])
        if block['class'] == 'FusionResample':
            scale_factor = block['scale_factor']
            in_channels = block['in_channels']
            out_channels = block['out_channels']
            mode = block.get('mode', 'nearest')
            return FusionResample(scale_factor, in_channels, out_channels, mode)
        if block['class'] == 'FusionMethod':
            inputs = block.get('inputs', number_track-1)
            out_channels = block.get('out_channels', number_channel*2)
            return FusionMethod(block['type'], in_channels=block['in_channels'], 
                                inputs=inputs,  out_channels=out_channels,
                                outputs=len(block['children']))
        if block['class'] == 'FusionClassification':
            return FusionClassification(block['input_channels'],block['classes'])
        if block['class'] == 'FusionOutput':
            return FusionOutput(block['inputs'])
        if block['class'] == 'PoolEncoder':
            output_size = block['output_size']
            mode = block.get('mode', 'max')
            return PoolEncoder(output_size, mode)
        if block['class'] == 'EncoderBlock':
            in_channels = block['in_channels']
            out_channels = block['out_channels']
            pool_mode = block.get('pool_mode', None)
            stride = block.get('stride', 2)
            return EncoderBlock(in_channels, out_channels, pool_mode, stride)
        if block['class'] == 'PreDecoderBlock':
            start_channels = block['start_channels']
            kernel_size = block['kernel_size']
            start_out_channels = block['start_out_channels']
            upsample_with_interpolation = block.get('upsample_with_interpolation', False)
            return PreDecoderBlock(start_channels, kernel_size, start_out_channels, upsample_with_interpolation)
        if block['class'] == 'DecoderBlock':
            in_channels = block['in_channels']
            out_channels = block['out_channels']
            pool_mode = block.get('pool_mode', None)
            stride = block.get('stride', 2)
            return DecoderBlock(in_channels, out_channels, pool_mode, stride)
        if block['class'] == 'PostDecoderBlock':
            in_channels = block['in_channels']
            return PostDecoderBlock(in_channels)
        if block['class'] == 'SpatialDecoder' or block['class'] == 'PoolDecoder':
            num_blocks = block['num_blocks']
            start_channels = block['start_channels']
            upsample_with_interpolation=block.get('upsample_with_interpolation', True)
            return SpatialDecoder(num_blocks, start_channels, upsample_with_interpolation)
        if block['class'] == 'FlatDecoder':
            num_blocks = block['num_blocks']
            start_channels = block['start_channels']
            start_out_channels = block.get('start_out_channels', 512)
            kernel_size = block.get('kernel_size', (7,14))
            upsample_with_interpolation=block.get('upsample_with_interpolation', False)
            return FlatDecoder(num_blocks, start_channels, start_out_channels, upsample_with_interpolation, kernel_size)
        if block['class'] == 'FCPreprocess':
            in_channels = block['in_channels']
            kernel_size = block['kernel_size']
            start_out_channels = block.get('start_out_channels', None)
            return FCPreprocess(in_channels, kernel_size, start_out_channels)
        if block['class'] == 'ConvPreprocess':
            in_channels = block['in_channels']
            kernel_size = block['kernel_size']
            start_out_channels = block.get('start_out_channels', None)
            return ConvPreprocess(in_channels, kernel_size, start_out_channels)
        if block['class'] == 'DropoutBlock':
            p = block.get('probability', 0.5)
            dims = block['dims']
            return DropoutBlock(p, dims)
        
    def forward(self, x):
        for layer in self.modulelist:
            for block in layer:
                x = block(x)
        return x

class FusionMethod(torch.nn.Module):
    def __init__(self, type, in_channels=0, inputs=0, out_channels=0, outputs = 0):
        super().__init__()
        assert type in ['add', 'conv', 'conv_per_output']
        self.outputs = outputs
        self.forward_fnc = self._add
        if type == 'add':
            self.fuse_layer = nn.BatchNorm2d(out_channels)
        if type == 'conv':
            self.fuse_layer = nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), bias=False),
                  nn.BatchNorm2d(out_channels)
                  )
            self.forward_fnc = self._conv
        if type == 'conv_per_output':
            self.fuse_layers = nn.ModuleList()
            for _ in range(outputs):
              self.fuse_layers.append(nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), bias=False),
                  nn.BatchNorm2d(out_channels)
                  ))
            self.forward_fnc = self._conv_per_output

        self.edges = []
        self.storage = [None]*inputs
        self.idx = 0

    def store_result(self, value):
        self.storage[self.idx] = value
        self.idx = (self.idx + 1)%(len(self.storage))

    def _add(self,x):
        x = torch.stack(x).sum(dim = 0)
        res = self.fuse_layer(x)
        for edgeObj in self.edges:
            edgeObj.store_result(res)

    def _conv(self,x):
        x = torch.hstack(x)
        res = self.fuse_layer(x)
        for edgeObj in self.edges:
            edgeObj.store_result(res)

    def _conv_per_output(self,x):
        x = torch.hstack(x)
        for i, layer in enumerate(self.fuse_layers):
            ret = layer(x)
            self.edges[i].store_result(ret)

    def forward(self, x):
        self.forward_fnc(self.storage)
        return

class FusionRoot(torch.nn.Module):
    def __init__(self):
        super().__init__()        
        self.edges = []

    def forward(self, x):
        for edgeObj in self.edges:
            edgeObj.store_result(x)
        return
class FusionRootWithStem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        stem = []
        stem.append(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        stem.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        stem.append(nn.ReLU(inplace=True))
        stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
        self.stem = nn.Sequential(*stem)
        self.edges = []
        self.storage = None
        
    def forward(self, x):
        res = self.stem(x)
        for edgeObj in self.edges:
            edgeObj.store_result(res)
        return

class FusionStem(torch.nn.Module):
    def __init__(self, num_blocks, stride=2, out_channels = 64):
        super().__init__()
        stem = []
        stem.append(nn.Conv2d(3, out_channels, kernel_size=(7, 7), stride=stride, padding=(3, 3), bias=False))
        stem.append(nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        stem.append(nn.ReLU(inplace=True))
        stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
        self.stem = nn.Sequential(*stem)

        modules = []
        for _ in range(0, num_blocks):
            modules.append(torchvision.models.resnet.BasicBlock(out_channels, out_channels))
        self.layers = nn.Sequential(*modules)

        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value
        
    def forward(self, x):
        res = self.stem(self.storage)
        res = self.layers(res)
        for edgeObj in self.edges:
            edgeObj.store_result(res)
        return

class FusionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        modules = []
        modules.append(
            torchvision.models.resnet.BasicBlock(in_channels, out_channels, stride=2, 
                                                 downsample=StandardDownsample(in_channels, out_channels, 
                                                 stride=2)))
        for _ in range(1, num_blocks):
            modules.append(torchvision.models.resnet.BasicBlock(out_channels, out_channels))
        self.layers = nn.Sequential(*modules)
        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value
        
    def forward(self, x):
        res = self.layers(self.storage)
        for edgeObj in self.edges:
            edgeObj.store_result(res)
        return
class FusionFirstLayer(torch.nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        modules = []
        for _ in range(0, num_blocks):
            modules.append(torchvision.models.resnet.BasicBlock(64, 64))
        self.layers = nn.Sequential(*modules)

        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value
        
    def forward(self, x):
        res = self.layers(self.storage)
        for edgeObj in self.edges:
            edgeObj.store_result(res)
        return

class FusionDownsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size =(1,1), stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value
        
    def forward(self, x):
        res = self.conv1(self.storage)
        res = self.bn1(res)
        for edgeObj in self.edges:
            edgeObj.store_result(res)
        return
class FusionDownsampleAvgPool(torch.nn.Module):
    def __init__(self, kernel_size=(1,1), stride=2, padding=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride, padding=padding)
        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value
        
    def forward(self, x):
        res = self.pool(self.storage)
        for edgeObj in self.edges:
            edgeObj.store_result(res)
        return

class FusionDownsampleMaxPool(torch.nn.Module):
    def __init__(self, kernel_size=(1,1), stride=2, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding=padding)
        self.edges = []
        self.storage = None
    
    def store_result(self, value):
        self.storage = value

    def forward(self, x):
        res = self.pool(self.storage)
        for edgeObj in self.edges:
            edgeObj.store_result(res)
        return

class FusionResample(torch.nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, mode='nearest'):
        super().__init__()
        pool = nn.Upsample(scale_factor=scale_factor, mode=mode)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3 , padding = 1, bias = False)
        bn = nn.BatchNorm2d(out_channels)
        self.edges = []
        self.storage = None
        self.block = nn.Sequential(pool, conv, bn)

    def store_result(self, value):
        self.storage = value
        
    def forward(self, _):
        res = self.block(self.storage)
        for edgeObj in self.edges:
            edgeObj.store_result(res)
        return

class FusionClassification(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value

    def forward(self, x):
        x = self.avgpool(self.storage)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        for edgeObj in self.edges:
            edgeObj.store_result(x)
        return 
class FusionOutput(torch.nn.Module):
    def __init__(self, inputs):
        super().__init__()        
        self.storage = [None]*inputs
        self.idx = 0

    def store_result(self, value):
        self.storage[self.idx] = value
        self.idx = (self.idx + 1)%(len(self.storage))

    def forward(self, _):
        return self.storage

class StandardDownsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size =(1,1), stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
    def forward(self, x):
        x = self.conv1(x)
        return self.bn1(x)

class PoolEncoder(torch.nn.Module):
    def __init__(self, output_size, mode = 'max'):
        super().__init__()
        if mode == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        elif mode == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        else:
            raise Exception( "mode arg can only be \'max\' or \'avg\'")
        self.edges = []
        self.storage = None
    
    def store_result(self, value):
        self.storage = value
        
    def forward(self, _):
        res = self.pool(self.storage)
        for edgeObj in self.edges:
            edgeObj.store_result(res)


class FCPreprocess(torch.nn.Module):
    def __init__(self, in_channels, kernel_size , start_out_channels = None):
        super().__init__()
        self.in_channels = in_channels
        self.start_out_channels = start_out_channels
        self.fc = nn.Identity()
        if start_out_channels is not None:
            self.fc = nn.Linear(in_channels * kernel_size[0] * kernel_size[1], start_out_channels)
            in_channels = start_out_channels
        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value

    def forward(self, _):
        x = self.storage
        x = x.flatten(1)
        x = self.fc(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        for edgeObj in self.edges:
            edgeObj.store_result(x)

class ConvPreprocess(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, start_out_channels = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        if start_out_channels:
            self.out_channels = start_out_channels
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size = kernel_size, bias = False)
        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value

    def forward(self, _):
        x = self.storage
        x = self.conv(x)
        for edgeObj in self.edges:
            edgeObj.store_result(x)



class FlatDecoder(torch.nn.Module):
    def __init__(self, num_blocks, start_channels, start_out_channels = 512, upsample_with_interpolation=False, kernel_size = (7,14)):
        assert start_out_channels * ((1/2)**num_blocks) > 3
        super().__init__()
        self.blocks = torch.nn.ModuleList()

        if upsample_with_interpolation:
            first_block = nn.Sequential(
                nn.Upsample(size = kernel_size),
                nn.Conv2d(start_channels, start_out_channels, kernel_size = (3,3), padding = (1,1), bias = False),
                nn.BatchNorm2d(start_out_channels),
                nn.ReLU())
        else:
            first_block = nn.Sequential(
                nn.ConvTranspose2d(start_channels, start_out_channels, kernel_size = kernel_size, 
                                    bias = False, stride = 1, padding = 0),
                nn.BatchNorm2d(start_out_channels),
                nn.ReLU())
        self.blocks.append(first_block)
        current_channels = start_out_channels
        for i in range(num_blocks):
            if upsample_with_interpolation:
                block = nn.Sequential(
                    nn.Upsample(scale_factor = 2),
                    nn.Conv2d(
                        in_channels = current_channels, 
                        out_channels = current_channels // 2, 
                        kernel_size=(3,3), padding=(1,1), bias=False),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU())
            else:
                block = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels = current_channels, 
                        out_channels = current_channels // 2, 
                        kernel_size=(4,4), stride = (2,2), padding=(1,1), bias=False),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU())
            self.blocks.append(block)
            current_channels //= 2
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels = current_channels, out_channels = 3, 
                kernel_size=(3,3), padding=(1,1), bias=False),
                nn.Sigmoid())
        )
        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value

    def forward(self,_):
        res = self.storage
        for block in self.blocks:
            res = block(res)
        for edgeObj in self.edges:
            edgeObj.store_result(res)
            
class SpatialDecoder(torch.nn.Module):
    def __init__(self, num_blocks, start_channels, upsample_with_interpolation = True):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        current_channels = start_channels
        for i in range(num_blocks):
            if upsample_with_interpolation:
                block = nn.Sequential(
                    nn.Upsample(scale_factor = 2),
                    nn.Conv2d(
                        in_channels = current_channels, 
                        out_channels = current_channels // 2, 
                        kernel_size=(3,3), padding=(1,1), bias=False),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU())
            else:
                block = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels = current_channels, 
                        out_channels = current_channels // 2, 
                        kernel_size=(4,4), stride = (2,2), padding=(1,1), bias=False),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU())
            self.blocks.append(block)
            current_channels //= 2
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels = current_channels, out_channels = 3, 
                kernel_size=(3,3), padding=(1,1), bias=False),
                nn.Sigmoid())
        )
        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value

    def forward(self,_):
        res = self.storage
        for block in self.blocks:
            res = block(res)
        for edgeObj in self.edges:
            edgeObj.store_result(res)

class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=3, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))

class Discriminator(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        n_D = 64
        self.blocks = nn.ModuleList()
        self.blocks.append(D_block(n_D))
        for _ in range(num_blocks - 1):
            block = D_block(n_D*2, n_D)
            self.blocks.append(block)
            n_D *= 2
        final_block = nn.Conv2d(in_channels=n_D, out_channels=1,
                    kernel_size=(7,14), bias=False)
        self.blocks.append(final_block)
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
        
class PreDecoderBlock(nn.Module):
    def __init__(self, start_channels, kernel_size, start_out_channels, upsample_with_interpolation=False):
        super().__init__()
        if upsample_with_interpolation:
            block = nn.Sequential(
                nn.Upsample(size = kernel_size),
                nn.Conv2d(start_channels, start_out_channels, kernel_size = (3,3), padding = (1,1), bias = False),
                nn.BatchNorm2d(start_out_channels),
                nn.ReLU())
        else:
            block = nn.Sequential(
                nn.ConvTranspose2d(start_channels, start_out_channels, kernel_size = kernel_size, 
                                    bias = False, stride = 1, padding = 0),
                nn.BatchNorm2d(start_out_channels),
                nn.ReLU())
        self.block = block

        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value
        
    def forward(self, _):
        x = self.storage
        x = self.block(x)
        for edge in self.edges:
            edge.store_result(x)

class PostDecoderBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels = 3, 
                kernel_size=(3,3), padding=(1,1), bias=False),
                nn.Sigmoid())

        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value
        
    def forward(self, _):
        x = self.storage
        x = self.block(x)
        for edge in self.edges:
            edge.store_result(x)

class DropoutBlock(nn.Module):
    def __init__(self, p, dims):
        super().__init__()
        if dims == 1:
            self.dropout = nn.Dropout(p)
        elif dims == 2:
            self.dropout = nn.Dropout2d(p)
        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value
        
    def forward(self, _):
        x = self.storage
        x = self.dropout(x)
        for edge in self.edges:
            edge.store_result(x)
            
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_with_interpolation = False, stride=2):
        super().__init__()
        if upsample_with_interpolation:
            block = nn.Sequential(
            nn.Upsample(scale_factor = stride),
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size=(3,3), padding=(1,1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        else:
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels = in_channels, 
                    out_channels = out_channels, 
                    kernel_size= 2 * stride, stride = stride, padding = stride//2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        
        self.block = block

        self.edges = []
        self.storage = None

    def store_result(self, value):
        self.storage = value
        
    def forward(self, _):
        x = self.storage
        x = self.block(x)
        #print(x.shape)
        for edge in self.edges:
            edge.store_result(x)
            
class EncoderBlock(nn.Module):
    # TODO parameterixe use of avg or maxpool
    def __init__(self, in_channels, out_channels, pool_mode = None, stride=2):
        super().__init__()
        conv_stride = 1 if pool_mode else stride
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = (3,3),
                                stride=conv_stride, padding = (1,1), bias=False)
        self.pool = {
            'max' : nn.MaxPool2d((2,2), 2),
            'avg' : nn.AvgPool2d((2,2), 2),
            None  : nn.Identity()
        }[pool_mode]

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        self.edges = []
        self.storage = None
    
    def store_result(self, value):
        self.storage = value

    def forward(self, _):
        res = self.activation(self.batch_norm(self.pool(self.conv2d(self.storage))))
        for edgeObj in self.edges:
            edgeObj.store_result(res)
        return

"""ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=197, bias=True)
)"""

def storage_use(model, batch_size=64, byte_size=2, unit='gb'):
    if unit not in ['gb', 'mb', 'kb', 'b']:
        print( 'unknown unit' )
    
    units = {'gb' : 3, 'mb' : 2, 'kb' : 1, 'k' : 0}

    model_size = 0
    for param in model.parameters():
        model_size += param.numel()
    model_size *= byte_size

    storage = []

    def pack(x):
        storage.append(x.numel())
        return len(storage) - 1

    def unpack(x):
        return storage[x]
        
    x =  torch.rand((batch_size,3,224,448))
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        _ = model(x)
    num_elm = torch.sum(torch.tensor(storage)).item()

    batch_usage = num_elm * byte_size

    return (model_size / 1024**units[unit]), (batch_usage / 1024**units[unit])