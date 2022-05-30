from IPython.display import clear_output
import graphviz;
clear_output()
import models
import fusion_model_dictionaries as fmd
import torch

def type_to_tag(module):
    return str(type(module)).split('.')[-1].split('\'')[0]

def size_to_str(size):
    if type(size) == int:
        return  str(size)
    ret = ""
    for d in size:
        ret += str(d)+ '\u00D7'
    return ret[:-1]




class base_node():
    cluster_name = 'cluster_'
    cluster_color = 'lightgrey'
    node_color = 'Silver'
    def __init__(self, module, gid, cluster_id):
        self.tag = type_to_tag(module)
        self.gid = gid
        self.cluster_id = str(cluster_id)
        self.obj_id = id(module)
        self.ref = module
        self.edges_out = []
        self.edge_count = 0
        self.storage = []
        if type(module.storage) == list:
            temp_list = []
            for i in range(len(module.storage)):
                temp_list.append(module.storage[i].shape[1:])
            self.storage = temp_list
        else:
            self.storage = [module.storage.shape[1:]]
        
    def get_edge_id(self):
        edge_name = str(self.gid)
        edge_channel = self.storage[self.edge_count]
        self.edge_count += 1
        return edge_name, edge_channel
    
    def create_label(self):
        mid = self.tag
        return mid 
    
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  label, shape='box',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)
            
    def draw_edges(self, gra):
        for i in range(len(self.edges)):
            gra.edge(str(self.gid) , str(self.gid) + '_' + self.edges[i], arrowhead='none',contraint='false')
            gra.node(str(self.gid) + '_' + self.edges[i],  size_to_str(self.edges_out[i]), height='0.3', shape='box',margin="0.05",style='rounded') 
            gra.edge(str(self.gid) + '_' + self.edges[i], self.edges[i],  contraint='false')
    
    def draw(self, gra):
        label = self.create_label()
        self.draw_node(gra, label)
        self.draw_edges(gra)
        
        
class fusion_node(base_node):
    def __init__(self, module, gid, cluster_id):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_f'
        self.cluster_id = str(gid)
        if module.forward_fnc.__name__ == '_conv_per_output':
            self.method = 'CPO'
        elif module.forward_fnc.__name__ == '_conv':
            self.method = 'Conv'
        else:
            self.method = 'Add'
            
        self.cluster_color = 'red'
        self.node_color = 'Crimson'
    
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  '{Fusion|' + self.method + '}', shape='record',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)    
            
class fusion_layer_node(base_node):
    def __init__(self, module, gid, cluster_id):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_l'
        self.blocks = len(module.layers)
        self.cluster_color = 'DodgerBlue'
        self.node_color = 'DeepSkyBlue'
    def create_label(self):
        if self.blocks == 1:
            return 'ResNet BasicBlock'
        else:
            return str(self.blocks) + ' ResNet BasicBlocks'

class fusion_classification_node(base_node):
    def __init__(self, module, gid, cluster_id):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_c'
        self.cluster_color = 'MediumSeaGreen'
        self.node_color = 'ForestGreen'
    def create_label(self):
        return 'Classification'
    
    def draw_edges(self, gra):
        for i in range(len(self.edges)):
            gra.edge(str(self.gid) , str(self.gid) + '_' + self.edges[i], contraint='false')
            gra.node(str(self.gid) + '_' + self.edges[i],  'Output\n' + size_to_str(self.edges_out[i]), shape='box', style='filled', color='skyblue') 

class decode_node(base_node):
    def __init__(self, module, gid, cluster_id, protype):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_decode'
        self.cluster_color = 'Orange'
        self.node_color = 'DarkOrange'
        self.cluster_id = str(gid)
        self.process_type = protype
        if module.blocks[0][0] == torch.nn.ConvTranspose2d:
            self.sample_type = 'Transposed Convolution'
        else:
            self.sample_type = 'Upsample interpolation'
            
        self.blocks = len(module.blocks)
        
    def draw_edges(self, gra):
        for i in range(len(self.edges)):
            gra.edge(str(self.gid) , str(self.gid) + '_' + self.edges[i], contraint='false')
            gra.node(str(self.gid) + '_' + self.edges[i],  'Output\n' + size_to_str(self.edges_out[i]), shape='box', style='filled', color='skyblue') 
            
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  '{' + self.process_type + ' Decoder|' + self.sample_type + '|' + str(self.blocks) +' blocks}', shape='record',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)   

class decode_start_block_node(base_node):
    def __init__(self, module, gid, cluster_id, start):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_decode'
        self.cluster_color = 'Orange'
        self.node_color = 'DarkOrange'
        self.cluster_id = str(gid)
        self.start = start
        if module.block[0] == torch.nn.ConvTranspose2d:
            self.sample_type = 'Transposed Convolution'
        else:
            self.sample_type = 'Upsample interpolation'
            
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  '{ Decoder ' + self.start + '| Single block}', shape='record',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)   
            
class decode_end_block_node(base_node):
    def __init__(self, module, gid, cluster_id):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_decode'
        self.cluster_color = 'Orange'
        self.node_color = 'DarkOrange'
        self.cluster_id = str(gid)
    def draw_edges(self, gra):
        for i in range(len(self.edges)):
            gra.edge(str(self.gid) , str(self.gid) + '_' + self.edges[i], contraint='false')
            gra.node(str(self.gid) + '_' + self.edges[i],  'Output\n' + size_to_str(self.edges_out[i]), shape='box', style='filled', color='skyblue')       
              
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  '{ Decoder Output| Single block}', shape='record',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)   
            
class root_node():
    
    def __init__(self, module, gid, cluster_id):
        self.tag = type_to_tag(module)
        self.gid = gid
        self.cluster_id = str(cluster_id)
        self.obj_id = id(module)
        self.ref = module
        self.edges_out = []
        self.input_size = []
        
    def draw(self, gra):
        gra.node(str(self.gid),  'Input\n' + size_to_str(self.edges_out[0]) , shape='box', style='filled', color='skyblue') 
        for i in range(len(self.edges)):
            gra.edge(str(self.gid) , self.edges[i], contraint='false')
            
class root_stem_node():
    def __init__(self, module, gid, cluster_id):
        self.tag = type_to_tag(module)
        self.gid = gid
        self.cluster_id = str(cluster_id)
        self.obj_id = id(module)
        self.ref = module
        self.edges_out = []
        self.input_size = []
        
    def draw(self, gra):
        gra.node(str(self.gid),  'Input\n' + size_to_str([3,224,448]) , shape='box', style='filled', color='skyblue') 
        gra.edge(str(self.gid) , str(self.gid) + '_stem', contraint='false')
        with gra.subgraph(name='cluster_stem' + self.cluster_id) as c:
            c.node(str(self.gid) + '_stem',  'Network Stem', shape='box',style='rounded,filled', fillcolor='DeepSkyBlue') 
            c.attr(style='rounded,filled', color='DodgerBlue')
        for i in range(len(self.edges)):
            gra.edge(str(self.gid) + '_stem' , str(self.gid) + '_' + self.edges[i], arrowhead='none',contraint='false')
            gra.node(str(self.gid) + '_' + self.edges[i],  size_to_str(self.edges_out[i]), height='0.3', shape='box',margin="0.05",style='rounded') 
            gra.edge(str(self.gid) + '_' + self.edges[i], self.edges[i],  contraint='false')
            
class stem_node(base_node):
    def __init__(self, module, gid, cluster_id):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_l'
        self.blocks = len(module.layers)
        self.cluster_color = 'DodgerBlue'
        self.node_color = 'DeepSkyBlue'
    def create_label(self):
        start = 'Network Stem'
        if self.blocks == 1:
            return start, 'ResNet BasicBlock'
        else:
            return start, str(self.blocks) + ' ResNet BasicBlocks'
        
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  '{' + label[0] + '|' + label[1] + '}', shape='record',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)    

class downsample_node(base_node):
    def __init__(self, module, gid, cluster_id, downtype):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_down'
        self.cluster_id = str(gid)
        self.cluster_color = 'MediumPurple'
        self.node_color = 'MediumOrchid'
        self.downtype = downtype
        
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  '{Downsample|' + self.downtype + '}', shape='record',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)    
            
class resample_node(base_node):
    def __init__(self, module, gid, cluster_id):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_down'
        self.cluster_id = str(gid)
        self.cluster_color = 'MediumPurple'
        self.node_color = 'MediumOrchid'
        
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  'Resample', shape='box',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)   

class pool_node(base_node):
    def __init__(self, module, gid, cluster_id):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_enc'
        self.cluster_id = str(gid)
        self.cluster_color = 'GoldenRod'
        self.node_color = 'Gold'
        if module.pool == torch.nn.AdaptiveMaxPool2d:
            self.pool_type = 'Max Pool'
        else:
            self.pool_type = 'Avg Pool'
        
        
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  '{Pool Encoder|' + self.pool_type + '}', shape='record',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)   
            
class fusion_encoder_block_node(base_node):
    def __init__(self, module, gid, cluster_id):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_enc'
        self.cluster_id = str(gid)
        self.cluster_color = 'GoldenRod'
        self.node_color = 'Gold'
        
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  '{ Encoder | Single block}', shape='record',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)   
            
class preprocess_node(base_node):
    def __init__(self, module, gid, cluster_id, protype):
        super().__init__(module, gid, cluster_id)
        self.cluster_name = 'cluster_enc'
        self.cluster_id = str(gid)
        self.cluster_color = 'GoldenRod'
        self.node_color = 'Gold'
        self.process_type = protype
        
    def draw_node(self, gra, label):
        with gra.subgraph(name=self.cluster_name+ self.cluster_id) as c:
            c.node(str(self.gid),  '{Preprocessing|' + self.process_type + '}', shape='record',style='rounded,filled', fillcolor=self.node_color) 
            c.attr(style='rounded,filled', color=self.cluster_color)               

def determ_node(module, gid, cluster_id):
    m_type = type_to_tag(module)
    if m_type == 'FusionRoot':
        return root_node(module, gid, cluster_id)
    if m_type == 'FusionRootWithStem':
        return root_stem_node(module, gid, cluster_id)
    if m_type == 'FusionStem':
        return stem_node(module, gid, cluster_id)
    if m_type == 'FusionMethod':
        return fusion_node(module, gid, cluster_id)
    if m_type == 'FusionClassification':
        return fusion_classification_node(module, gid, cluster_id)
    if m_type == 'SpatialDecoder':
        return decode_node(module, gid, cluster_id,'Spatial')
    if m_type == 'FlatDecoder':
        return decode_node(module, gid, cluster_id, 'Flat')
    if m_type == 'PostDecoderBlock':
        return decode_end_block_node(module, gid, cluster_id)
    if m_type == 'PreDecoderBlock':
        return decode_start_block_node(module, gid, cluster_id, 'Input')
    if m_type == 'DecoderBlock':
        return decode_start_block_node(module, gid, cluster_id, '')
    if m_type == 'FusionLayer' or m_type == 'FusionFirstLayer':
        return fusion_layer_node(module, gid, cluster_id)
    if m_type == 'EncoderBlock':
        return fusion_encoder_block_node(module, gid, cluster_id)
    if m_type == 'FusionDownsample':
        return downsample_node(module, gid, cluster_id, 'Stride')
    if m_type == 'FusionDownsampleAvgPool':
        return downsample_node(module, gid, cluster_id, 'Avg Pool')
    if m_type == 'FusionDownsampleMaxPool':
        return downsample_node(module, gid, cluster_id, 'Max Pool')
    if m_type == 'FusionResample':
        return resample_node(module, gid, cluster_id)
    if m_type == 'PoolEncoder':
        return pool_node(module, gid, cluster_id)
    if m_type == 'FCPreprocess':
        return preprocess_node(module, gid, cluster_id, 'Fully connected')
    if m_type == 'ConvPreprocess':
        return preprocess_node(module, gid, cluster_id,'Convolution')

    return base_node(module, gid, cluster_id)

class graph():    
    def __init__(self, model):
        model(torch.rand((1,3,224,448)))
        self.digraph = []
        gid = 0
        cluster_id = 0
        for outer_module in model.modulelist:
            for inner_module in outer_module:
                node = determ_node(inner_module, gid, cluster_id)
                gid += 1
                self.digraph.append(node)
            cluster_id += 1
        for i in range(len(self.digraph)):
            edges = []
            edges_channels = []
            ref_edges = self.digraph[i].ref.edges
            
            for j in range(len(ref_edges)):
                for k in range(i+1, len(self.digraph)):
                    if id(ref_edges[j]) == self.digraph[k].obj_id:
                        edge_id, edge_channels = self.digraph[k].get_edge_id()
                        edges.append(edge_id)
                        edges_channels.append(edge_channels)
                        
            self.digraph[i].edges = edges
            self.digraph[i].edges_out = edges_channels
        self.digraph = self.digraph[:-1]
    def draw(self, file_name, path='fusion_graphs/', direction = 'TD'):
        gra = graphviz.Digraph(file_name, engine='dot',graph_attr={'rankdir': direction})
        for i in range(len(self.digraph)):
            self.digraph[i].draw(gra)
        gra.render(path + file_name, format='png')