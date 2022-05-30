import PIL.Image as Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import io
from ipywidgets import widgets
from PIL import Image
import json
import models
from PIL import Image

def epoch_printout(metrics, best_metrics, taxa, recon_loss_num_elems = 0):
    titles  = '{:<15}'.format('taxon:')
    losses  = '{:<15}'.format('loss:')
    accs    = '{:<15}'.format('acc:')
    best_loss   = '{:<15}'.format('best loss:')
    best_acc   = '{:<15}'.format('best acc:')
    #training stats
    train_strings = 'Training stats:\n\n'
    for i in range(len(taxa)):
        titles  += '{:<13}'.format(taxa[i])
        losses  += '{:<13.4f}'.format(metrics[0]['train'][i])
        accs    += '{:<13.4f}'.format(metrics[1]['train'][i])
        best_loss  += '{:<13.4f}'.format(best_metrics[0]['train'][i])
        best_acc   += '{:<13.4f}'.format(best_metrics[1]['train'][i])
    train_strings += titles + '\n' + losses  + '\n' +  accs  + '\n' + best_loss + '\n' + best_acc + '\n\n'
    titles  = '{:<15}'.format('taxon:')
    losses  = '{:<15}'.format('loss:')
    accs    = '{:<15}'.format('acc:')
    best_loss   = '{:<15}'.format('best loss:')
    best_acc   = '{:<15}'.format('best acc:')
    val_strings = 'Validation stats:\n\n'
    for i in range(len(taxa)):
        titles  += '{:<13}'.format(taxa[i])
        losses  += '{:<13.4f}'.format(metrics[0]['val'][i])
        accs    += '{:<13.4f}'.format(metrics[1]['val'][i])
        best_loss  += '{:<13.4f}'.format(best_metrics[0]['val'][i])
        best_acc   += '{:<13.4f}'.format(best_metrics[1]['val'][i])
    val_strings += titles + '\n' + losses  + '\n' +  accs  + '\n' + best_loss + '\n' + best_acc + '\n\n'
    
    if recon_loss_num_elems != 0:
        recon_train_loss = ''
        recon_val_loss = ''
        recon_loss_labels = ['recon_loss', 'adv_loss', 'disc_loss']
        title = '{:<13} {:<13} {:<13}\n'.format('extras:', 'current', 'best')
        for i in range(len(taxa), len(taxa) + recon_loss_num_elems):
            recon_train_loss  += '{:<13} {:<13.4f} {:<13.4f}\n'.format(recon_loss_labels[i - len(taxa)], metrics[0]['train'][i], best_metrics[0]['train'][i])
            recon_val_loss += '{:<13} {:<13.4f} {:<13.4f}\n'.format(recon_loss_labels[i - len(taxa)], metrics[0]['val'][i], best_metrics[0]['val'][i])
        train_strings += title + recon_train_loss
        val_strings += title + recon_val_loss

    total_train_loss = '{:<13} {:<13.4f} {:<13.4f}\n'.format('total loss', metrics[0]['train'][-1], best_metrics[0]['train'][-1])
    total_val_loss = '{:<13} {:<13.4f} {:<13.4f}\n'.format('total loss', metrics[0]['val'][-1], best_metrics[0]['val'][-1])
    train_strings += total_train_loss + '\n'
    val_strings += total_val_loss
            
    return train_strings + val_strings

def plot_training_stats(metrics, taxa, additional, figsize=(16,9), dpi=200, path_prefix=None, ext = '.png', render_name='render_temp.png'):
    xs = metrics['epochs']
    taxa_aug= taxa if len(taxa) == 1 else taxa + ['total']
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    cmap = {'subfamily': 'tab:blue', 'tribe': 'tab:orange', 
                 'genus': 'tab:green', 'species':'tab:red', 
                 'total': 'purple'}
    # subplots_adjust() is necessary to get proper spacing. plt.tight_layout() is unreliable and the constrained_layout param for plt.subplots() does not work when we try to extract individual subfigures
    plt.subplots_adjust(hspace=.3)
    for ax in axs.flat:
        ax.set_xlabel('Epochs')
        ax.grid()
    for i in range(2):
        axs[0,i].set_ylabel('Loss')
        axs[1,i].set_ylabel('Accuracy')
        axs[i, 0].set_title('Training')
        axs[i, 1].set_title('Validation')
    for i, loss_type in enumerate(['loss', 'acc']):
        for j, phase in enumerate(['train', 'val']):
            for category in taxa_aug:
                ys = metrics[category][phase][loss_type]
                axs[i, j].plot(xs, ys, label=category, color = cmap[category])
            axs[i, j].legend()
            if path_prefix is not None:
                extent = axs[i, j].get_tightbbox(fig.canvas.get_renderer()).transformed(
                    fig.dpi_scale_trans.inverted())
                save_path = path_prefix + '_' + loss_type + '_' + phase + ext
                # we expand the bounding box defined by "extent" to get the full subfigure
                fig.savefig(save_path, bbox_inches=extent.expanded(1.1,1.1), dpi=dpi, facecolor='w')
    #plt.show()
    fig.savefig(render_name, dpi=dpi, facecolor='w', bbox_inches='tight')
    plt.close()


def plot_training_stats_2(metrics, taxa, additional, figsize=(16,10), dpi=200, path_prefix=None, ext = '.png', render_name='render_temp.png'):
    xs = metrics['epochs']
    taxa_aug= taxa if len(taxa) == 1 else taxa + ['total']
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    cmap = {'subfamily': 'tab:blue', 'tribe': 'tab:orange', 
                 'genus': 'tab:green', 'species':'tab:red', 
                 'total': 'purple'}
    # subplots_adjust() is necessary to get proper spacing. plt.tight_layout() is unreliable and the constrained_layout param for plt.subplots() does not work when we try to extract individual subfigures
    plt.subplots_adjust(hspace=.3)
    for ax in axs.flat:
        ax.set_xlabel('Epochs')
        ax.grid()

    y_points = np.array([0,0.5,0.7,0.8,0.85,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99, 1])
    y_ticks = y_points**4
    y_labels = ["{0:.2f}".format(x) for x in y_points]
    for i in range(2):
        axs[0,i].set_ylabel('Loss')
        axs[1,i].set_ylabel('Accuracy')
        axs[i, 0].set_title('Training')
        axs[i, 1].set_title('Validation')
        axs[1,i].set_ylim([-0.025,1.025])
        axs[1,i].set_yticks(y_ticks)
        axs[1,i].set_yticklabels(y_labels, fontsize=7)


   
    for j, phase in enumerate(['train', 'val']):
        for category in taxa_aug:
            ys = metrics[category][phase]['loss']
            axs[0, j].plot(xs, ys, label=category, color = cmap[category])
        axs[0, j].legend()
        if path_prefix is not None:
            extent = axs[0, j].get_tightbbox(fig.canvas.get_renderer()).transformed(
                fig.dpi_scale_trans.inverted())
            save_path = path_prefix + '_loss_' + phase + ext
            # we expand the bounding box defined by "extent" to get the full subfigure
            fig.savefig(save_path, bbox_inches=extent.expanded(1.1,1.1), dpi=dpi, facecolor='w')

    for j, phase in enumerate(['train', 'val']):
        for category in taxa_aug:
            ys = np.array(metrics[category][phase]['acc'])
            axs[1, j].plot(xs, ys**4, label=category, color = cmap[category], alpha=0.6)
            best=np.argmax(ys)
            axs[1, j].scatter(xs[best], ys[best]**4, color = cmap[category])
        axs[1, j].legend()
        if path_prefix is not None:
            extent = axs[1, j].get_tightbbox(fig.canvas.get_renderer()).transformed(
                    fig.dpi_scale_trans.inverted())
            save_path = path_prefix + '_acc_' + phase + ext
            # we expand the bounding box defined by "extent" to get the full subfigure
            fig.savefig(save_path, bbox_inches=extent.expanded(1.1,1.1), dpi=dpi, facecolor='w')

    
    #plt.show()
    fig.savefig(render_name, dpi=dpi, facecolor='w', bbox_inches='tight')
    plt.close()

def plot_training_stats_3(metrics, taxa, additional, figsize=(16,15), dpi=200, path_prefix=None, ext = '.png', render_name='render_temp.png'):
    xs = metrics['epochs']
    taxa_aug= taxa if len(taxa) == 1 else taxa + ['total']
    fig, axs = plt.subplots(3, 2, figsize=figsize)
    cmap = {'subfamily': 'tab:blue', 'tribe': 'tab:orange', 
                 'genus': 'tab:green', 'species':'tab:red', 
                 'total': 'purple'}
    # subplots_adjust() is necessary to get proper spacing. plt.tight_layout() is unreliable and the constrained_layout param for plt.subplots() does not work when we try to extract individual subfigures
    plt.subplots_adjust(hspace=.3)
    for ax in axs.flat:
        ax.set_xlabel('Epochs')
        ax.grid()

    y_points = np.array([0,0.5,0.7,0.8,0.85,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99, 1])
    y_ticks = y_points**4
    y_labels = ["{0:.2f}".format(x) for x in y_points]
    for i in range(2):
        axs[0,i].set_ylabel('Loss')
        axs[1,i].set_ylabel('Accuracy')
        axs[i, 0].set_title('Training')
        axs[i, 1].set_title('Validation')
        axs[1,i].set_ylim([-0.025,1.025])
        axs[1,i].set_yticks(y_ticks)
        axs[1,i].set_yticklabels(y_labels, fontsize=7)

    train_spe_label = additional[1]['train'][-1].astype(int)
    val_spe_label = additional[1]['val'][-1].astype(int)
    train_spe_pred = additional[0]['train'][-1].astype(int)
    val_spe_pred =  additional[0]['val'][-1].astype(int)

    train_spe_pred = np.bincount(train_spe_label[train_spe_pred==train_spe_label],minlength=197)
    val_spe_pred = np.bincount(val_spe_label[val_spe_pred==val_spe_label],minlength=197)
    train_spe_label = np.bincount(train_spe_label,minlength=197)
    val_spe_label = np.bincount(val_spe_label,minlength=197)


   
    for j, phase in enumerate(['train', 'val']):
        for category in taxa_aug:
            ys = metrics[category][phase]['loss']
            axs[0, j].plot(xs, ys, label=category, color = cmap[category])
        axs[0, j].legend()
        if path_prefix is not None:
            extent = axs[0, j].get_tightbbox(fig.canvas.get_renderer()).transformed(
                fig.dpi_scale_trans.inverted())
            save_path = path_prefix + '_loss_' + phase + ext
            # we expand the bounding box defined by "extent" to get the full subfigure
            fig.savefig(save_path, bbox_inches=extent.expanded(1.1,1.1), dpi=dpi, facecolor='w')

    for j, phase in enumerate(['train', 'val']):
        for category in taxa_aug:
            ys = np.array(metrics[category][phase]['acc'])
            axs[1, j].plot(xs, ys**4, label=category, color = cmap[category], alpha=0.6)
            best=np.argmax(ys)
            axs[1, j].scatter(xs[best], ys[best]**4, color = cmap[category])
        axs[1, j].legend()
        if path_prefix is not None:
            extent = axs[1, j].get_tightbbox(fig.canvas.get_renderer()).transformed(
                    fig.dpi_scale_trans.inverted())
            save_path = path_prefix + '_acc_' + phase + ext
            # we expand the bounding box defined by "extent" to get the full subfigure
            fig.savefig(save_path, bbox_inches=extent.expanded(1.1,1.1), dpi=dpi, facecolor='w')

    axs[2,0].set_xlabel('Species')
    axs[2,0].set_ylabel('Samples')
    axs[2,0].plot(train_spe_label, label='dataset', color = 'tab:blue')
    axs[2,0].plot(train_spe_pred, label='correct samples', color = 'tab:red')
    axs[2,0].legend()
    axs[2,1].set_xlabel('Species')
    axs[2,1].set_ylabel('Samples')
    axs[2,1].plot(val_spe_label, label='dataset', color = 'tab:blue')
    axs[2,1].plot(val_spe_pred, label='correct samples', color = 'tab:red')
    axs[2,1].legend()

    
    #plt.show()
    fig.savefig(render_name, dpi=dpi, facecolor='w', bbox_inches='tight')
    plt.close()
class Rendering():
    """
        Class for rendering dreamt images.
    """

    def __init__(self, draw_function = None, shape =(400,800), scale = 2) -> None:
        self.format = 'png'
        if isinstance(shape, int):
            h, w =  (shape, shape)
        else:
            h, w = shape

        self.draw_function = draw_function
        start_image = np.full((h,w,3), 255).astype(np.uint8)
        image_stream = self.compress_to_bytes(start_image)
        self.image = widgets.Image(value = image_stream, width=w*scale, height=h*scale)
        self.image_box = widgets.Box([self.image])
        self.image_box.layout = widgets.Layout(display='flex',
                width='80%',
                border='solid 1px black',
                margin='10px 10px 10px 10px',
                padding='5px 5px 5px 5px')
        
        self.console_lines = [widgets.HTML('<body style="font-size:10vw"></body>') for i in range(20)]
        self.line_idx = 0
        self.console_column = widgets.VBox(self.console_lines)
        self.console_column.layout = widgets.Layout(display='flex',
                flex_flow='column',
                align_items='flex-start',
                width='80%',
                border='solid 1px black',
                margin='10px 10px 10px 10px',
                padding='5px 5px 5px 5px')
        
        self.output = widgets.HBox([self.image_box, self.console_column])
        self.output.layout = widgets.Layout(display='flex',
                flex_flow='column',
                align_items='flex-start')
        display(self.output) 


    # To display the images, they need to be converted to a stream of bytes
    def compress_to_bytes(self, data) -> bytes:
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        """
            Helper function to compress image data via PIL/Pillow.
        """
        buff = io.BytesIO()
        img = Image.fromarray(data)    
        img.save(buff, format=self.format)
    
        return buff.getvalue()

    def update(self, params = None, filename = None):
        if self.draw_function is not None and params is not None:
            if filename is None:
                filename = 'render_temp.png'
            self.draw_function(*params, render_name=filename)
        if filename is not None:
            img = np.array(Image.open(filename))
            self.update_(img)

    def update_(self, image ) -> None:
        stream = self.compress_to_bytes(image)
        time.sleep(1/144)
        self.image.value = stream

    def print_lines(self, lines):
        for i in range(len(self.console_lines)):
            self.console_lines[i].value = '<pre>' + str(lines[i:i+1])[2:-2] + '</pre>'
    
    def print(self, line):
        print_line = '<pre>' + line + '</pre>'
        self.console_lines[self.line_idx].value = print_line
        if self.line_idx < len(self.console_lines) - 1:
            self.line_idx += 1
        
    def clear_console(self):
        for i in range(len(self.console_lines)):
            self.console_lines[i].value = '<pre></pre>'
        self.line_idx = 0

def one_shot(model_name, path_prefix, taxa, device, transform, img_path):
    tmap = {'subfamily': '0', 'tribe': '1', 'genus': '2', 'species': '3'}
    model_tags = model_name.split(' ')
    model_name_ext = model_tags[0] + '_' + ''.join([tmap[taxon] for taxon in taxa])
    model_path = path_prefix + model_name_ext + ' ' + ' '.join(model_tags[1:])
    with open(model_path + '/metrics.json') as f:
        metrics = json.load(f)
    model_constructor = eval(metrics['model_type'][8:-2])
    aux_model_args = metrics['aux_model_args']
    constructor_args = {**aux_model_args, **
                        {'classes_per_taxon': metrics['classes_per_taxon']}}
    model = model_constructor(**constructor_args).to(device)
    model.eval()
    state_dict = torch.load(model_path + '/state_dict.pt')
    model.load_state_dict(state_dict)
    
    img = Image.open(img_path)
    input = transform[0](img).to(device)
    outputs = model(input.unsqueeze(0))
    recon = outputs[0].squeeze(0).cpu().detach().numpy().transpose((1, 2, 0))
    plt.figure(figsize=(10, 20))
    plt.imshow(np.array(img.resize((224, 112))))
    plt.figure(figsize=(10, 20))
    plt.imshow(recon)
