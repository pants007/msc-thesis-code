
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tb_to_metrics(model_name, taxa):
    path = '../output/msc-thesis-22/tensorboard/'
    global_tags = ['all_train_loss', 'all_val_loss', 'all_train_acc', 'all_val_acc']
    recon_tags = ['recon_loss', 'adv_loss', 'disc_loss']
    tmap = {'subfamily': '0', 'tribe': '1', 'genus': '2', 'species': '3'}
    model_tags = model_name.split(' ')
    model_name_ext = model_tags[0] + '_' + ''.join([tmap[taxon] for taxon in taxa])

    model_path_suffix = model_name_ext + '/' + '/'.join(model_tags[1:]) #need this for finding the model in general
    event_prefix = model_name_ext + '_' + '_'.join(model_tags[1:]) #need this for finding event folders for individual scalar series
    model_path = path + model_path_suffix

    metrics = {}
    #process event files (images) in base folder
    event_acc = EventAccumulator(model_path)
    event_acc.Reload()

    #process event files in subfolders
    last_one = 0 #holds index of last recorded step=1 in the ScalarEvents
    for taxon in taxa:
        metrics[taxon] = {}
        metrics[taxon]['val'] = {}
        metrics[taxon]['val']['acc'] = {}
        metrics[taxon]['val']['loss'] = {}
        metrics[taxon]['train'] = {}
        metrics[taxon]['train']['acc'] = {}
        metrics[taxon]['train']['loss'] = {}
        for tag in global_tags:
            tag_split = tag.split('_')
            path =  model_path + '/' + event_prefix + '_' + '_'.join(tag_split[1:]) + '_' + taxon
            event_acc = EventAccumulator(path)
            event_acc.Reload()
            all_events = event_acc.Scalars(event_prefix + '/' + '/'.join(tag_split[1:]))

            #find last recorded step=1 in case of previous partial training
            steps = [event.step for event in all_events]
            last_one = len(steps) - np.argmin(steps[::-1]) - 1
            actual_events = all_events[last_one:]
            epochs = [e.step for e in actual_events]
            metrics[taxon][tag_split[1]][tag_split[2]] = [e.value for e in actual_events]
            metrics['epochs'] = epochs
    for loss_name in recon_tags:
        metrics[loss_name] = {}
        metrics[loss_name]['train'] = {}
        metrics[loss_name]['val'] = {}
        for tag in global_tags[:2]:
            tag_split = tag.split('_')
            path =  model_path + '/' + event_prefix + '_' + '_'.join(tag_split[1:]) + '_' + loss_name
            try:
                event_acc = EventAccumulator(path)
                event_acc.Reload()
                all_events = event_acc.Scalars(event_prefix + '/' + '/'.join(tag_split[1:]))
                actual_events = all_events[last_one:]
                metrics[loss_name] = [e.value for e in actual_events]
            except Exception as e:
                print('{} not recorded. please verify :^)'.format(loss_name))
    return metrics