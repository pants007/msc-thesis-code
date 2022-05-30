from graphviz import Digraph
import torch
import collections.abc
import numpy as np
import json

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

class Graph_node:
    def __init__(self):
        self.nr = 0
        self.taxa_lvl = None
        self.taxa_idx = -1
        self.idx = 0
        self.name = ""
        self.parent = -1
        self.children = []
        self.acc = 0
        self.is_leaf = False
        self.dist = 0
        self.samples = 0
        self.correct = np.array([False])
        self.paths = []
        self.pos = []
        self.label = ''
    def __repr__(self):
        return self.name + '_node'

class Graph:
    def __init__(self, dataframe, graph_name):
        self.pos = np.load("./graphs/coords.npy")
        with open('./graphs/node_attr.json') as file:
            self.node_attr = json.load(file)
        with open('./graphs/inv_labels.json') as file:
            self.inv_dict = json.load(file)    
        taxa = ['subfamily', 'tribe', 'genus', 'species']
        self.taxa = taxa
        self.graph_name = graph_name
        self.label = self.std_label
        root_node = Graph_node()
        root_node.name = 'root'
        root_node.label = 'root'
        root_node.children = np.arange(len(self.inv_dict[taxa[0]]))
        root_node.pos = self.pos[0]
        self.graph = [[root_node]]
        node_nr = 1


        for i in range(len(taxa)):
            taxa_nodes = []
            for j in range(len(self.inv_dict[taxa[i]])):
                temp_node = Graph_node()
                temp_node.nr = node_nr
                temp_node.taxa_lvl = taxa[i]
                temp_node.taxa_idx = i
                temp_node.name = self.inv_dict[taxa[i]][str(j)]
                temp_node.idx = j
                temp_node.pos = self.pos[node_nr]

                g, mean = get_accuracy(dataframe, taxa[i], j)
                temp_node.acc = mean

                if i != 0:
                    parent_col = taxa[i-1] + '_label'
                    parent = np.unique(g[parent_col].to_numpy())
                    assert parent.shape[0] == 1
                    temp_node.parent = int(parent)

                if i != len(taxa) - 1:
                    child_col = taxa[i+1] + '_label'
                    children = np.unique(g[child_col].to_numpy())
                    temp_node.children = children
                else:
                    temp_node.is_leaf = True
                    temp_node.paths =  g['path'].to_numpy()
                _, dist = get_dists(dataframe, taxa[i], j)
                dist = dist.to_numpy()
                temp_node.dist = dist
                temp_node.samples = dist.shape[0]
                temp_node.correct = dist == 0

                temp_node.label = str(j)

                taxa_nodes.append(temp_node)
                node_nr += 1
            self.graph.append(taxa_nodes)

    def move_subtree(self, taxa, tree, delta):
        curr_taxa = taxa + 1
        nodes = [tree]
        for i in range(curr_taxa,len(self.graph)):
            new_nodes = []
            for j in nodes:
                self.graph[i][j].pos += delta

                if len(self.graph[i][j].children):
                    new_nodes += self.graph[i][j].children.tolist()
            nodes = new_nodes

    def rotate_subtree(self, taxa, tree, delta):
        curr_taxa = taxa + 1
        nodes = [tree]
        center = self.graph[curr_taxa][tree].pos
        rot = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])
        for i in range(curr_taxa,len(self.graph)):
            new_nodes = []
            for j in nodes:
                new_pos = self.graph[i][j].pos - center
                new_pos = np.dot(rot, new_pos)
                self.graph[i][j].pos = new_pos + center

                if len(self.graph[i][j].children):
                    new_nodes += self.graph[i][j].children.tolist()
            nodes = new_nodes

    def save_pos(self):
        idx = 0
        for i in range(len(self.graph)):
            for j in range(len(self.graph[i])):
                self.pos[idx] = self.graph[i][j].pos
                idx += 1

        np.save("./graphs/coords.npy", self.pos)

    def save_attr(self):
        with open('./graphs/node_attr.json', 'w') as file:
            json.dump(self.node_attr, file, indent = 4)   

    def std_label(self, node):
        if node.label != 'root':
            return "(" + node.label + ')\n' + str(np.sum(node.correct )) + '/' + str(node.samples)
        else:
            return 'root'

    def set_label_func(self, func):
        self.label = func

    def comp_label(self, node, comp_node):
        if node.label != 'root':
            return "(" + node.label + ')\n' + str(np.sum(node.correct[np.invert(comp_node.correct)])) + '/' + str(np.sum(comp_node.correct[np.invert(node.correct)])) + '\n' + str(np.sum(node.correct )) + '/' + str(node.samples)
        else:
            return 'root'

    

    def point(self, taxa, node):
        return str(self.graph[taxa][node].pos[1]) + ',' + str(self.graph[taxa][node].pos[0]) + '!'

    def acc_color(self, taxa, node):
        r = self.graph[taxa][node].acc * 255
        r_hex = hex(int(r))
        if len(r_hex) < 4:
            r_hex = '0' + r_hex[2:]
        else:
            r_hex = r_hex[2:]
        return '#ff' + r_hex +  r_hex

    def comp_color(self, node, comp_node):
        curr = np.sum(node.correct[np.invert(comp_node.correct)])
        comp = np.sum(comp_node.correct[np.invert(node.correct)])
        if curr == 0 and comp != 0:
            return'#00ff00'
        if curr != 0 and comp == 0:
            return '#0000ff'
        if curr != 0 and comp != 0:
            return '#ff0000'
        return '#ffffff'

    
    def idx_str(self, i,j):
        return str(i) + ',' + str(j)

    def render_complete_graph(self, size=None, format='png', comp=None):
        if size is not None:
            gra = Digraph(self.graph_name, engine='neato', graph_attr={'size':str(size)})
        else:
            gra = Digraph(self.graph_name, engine='neato')
        
        for i in range(len(self.graph)):
            with gra.subgraph() as t:
                t.attr(rank='same')
                for j in range(len(self.graph[i])):
                    attr = self.node_attr[str(i)].copy()
                    attr['pos'] = self.point(i,j)
                    if comp is not None:
                        attr['label'] = self.comp_label(self.graph[i][j], comp.graph[i][j])
                        attr['fillcolor'] = self.comp_color(self.graph[i][j], comp.graph[i][j])
                        attr['fontsize'] = str(int(attr['fontsize'])*0.8)
                    else:
                        attr['label'] = self.label(self.graph[i][j])
                        attr['fillcolor'] = self.acc_color(i,j)
                    t.node(self.idx_str(i,j),  _attributes=attr)
                    if len(self.graph[i][j].children):
                        for k in self.graph[i][j].children:
                            t.edge(self.idx_str(i,j),self.idx_str(i+1,k), constraint='True', _attributes={'penwidth' : '5'})

        for i in range(len(self.taxa)):
            attr = {}
            attr['label'] = self.taxa[i]
            attr['shape'] = self.node_attr[str(i+1)]['shape']
            attr['fixedsize'] = 'true'
            attr['height'] = '4'
            attr['width'] = '4'
            attr['fontsize'] = '60'
            attr['pos'] = str(i*8 +155) + ',60!'
            attr['penwidth'] = '5'
            gra.node("legend" + str(i), _attributes=attr)
        if comp is not None:
            filename = 'comp_' + self.graph_name + '_' + comp.graph_name
        else:
            filename = self.graph_name
        gra.render(filename = filename, directory = './../output/msc-thesis-22/hierarchy_graphs/'+filename + '/', format=format)

    def render_subtree(self, start_taxa, start_node, size=None, format='png', engine='circo',comp=None):
        if size is not None:
            gra = Digraph(self.graph_name, engine=engine, graph_attr={'size':str(size)})
        else:
            gra = Digraph(self.graph_name, engine=engine)

        nodes = [start_node]
        for i in range(start_taxa, len(self.graph)):
            with gra.subgraph() as t:
                t.attr(rank='same')
                new_nodes = []
                for j in nodes:
                        attr = self.node_attr[str(i)].copy()
                        if comp is not None:
                            attr['label'] = self.comp_label(self.graph[i][j], comp.graph[i][j])
                            attr['fillcolor'] = self.comp_color(self.graph[i][j], comp.graph[i][j])
                            attr['fontsize'] = str(int(attr['fontsize'])*0.8)
                        else:
                            attr['label'] = self.label(self.graph[i][j])
                            attr['fillcolor'] = self.acc_color(i,j)
                        t.node(self.idx_str(i,j),  _attributes=attr)
                        if len(self.graph[i][j].children):
                            for k in self.graph[i][j].children:
                                t.edge(self.idx_str(i,j),self.idx_str(i+1,k), _attributes={'penwidth' : '5'})
                            new_nodes += self.graph[i][j].children.tolist()
                nodes = new_nodes
        if comp is not None:
            filename = 'comp_' + self.graph_name + '_' + comp.graph_name
        else:
            filename = self.graph_name
        gra.render(filename = filename + '_subtree_' + str(start_taxa) + '_' + str(start_node), directory = './../output/msc-thesis-22/hierarchy_graphs/'+filename+ '/', format=format)
class Comp_graph:
    def __init__(self):
        pass

class Network_graph:
    def __init__(self):
        pass