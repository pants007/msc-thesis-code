def make_backward_graph_dict(fusion_type):
    graph_dict = {
    'root': {
        'name' : 'backward_graph_dict',
        'a' : {
        'class' : 'FusionRoot',
        'children' : ['stem:a','stem:b','stem:c','stem:d']
        }
    },
    'stem': {
        'a' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion', 'class:a', 'layer1:a', 'layer2:a']
            },
        'b' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'c' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'd' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 64 * 4,
            'type': fusion_type,
            'children': ['layer1:b','layer1:c','layer1:d']
            }
        },
    'layer1': {
        'a' : {
            'class': 'FusionDownsample',
            'input_channels' : 64,
            'stride' : 2,
            'children': ['layer1:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion', 'class:b', 'layer2:b']
            },
        'c' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'd' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 128 * 4,
            'type': fusion_type,
            'children': ['layer2:c', 'layer2:d']
            }
        },
    'layer2': {
        'a' : {
            'class': 'FusionDownsample',
            'input_channels' : 64,
            'stride' : 4,
            'children': ['layer2:fusion']
            },
        'b' : {
            'class': 'FusionDownsample',
            'input_channels' : 128,
            'stride' : 2,
            'children': ['layer2:fusion']
            },
        'c' : { 
            'class': 'FusionLayer', 
            'blocks' : 1,
            'children': ['layer2:fusion', 'class:c'] #TODO add child 'class:c' and remove correspondingly below
            },
        'd' : { 
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 256 * 4,
            'type': fusion_type,
            'children': ['layer3:d']
            }
        },
    'layer3': {
        'd' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:d']
            },
        },
    'class': {
        'a' : {
            'class': 'FusionClassification',
            'input_channels' : 64,
            'classes' : 2,
            'children': ['output:a']
            },
        'b' : {
            'class': 'FusionClassification',
            'input_channels' : 128,
            'classes' : 4,
            'children': ['output:a']
            },
        'c' : {
            'class': 'FusionClassification',
            'input_channels' : 256,
            'classes' : 44,
            'children': ['output:a']
            },
        'd' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 197,
            'children': ['output:a']
            },
        },
    'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 4,
            'children' : []
            }
        }
    }
    return graph_dict

def make_backward_graph_dict_interpolate(fusion_type, interpolation_type, interpolation_mode='nearest'):
    graph_dict = {
    'root': {
        'name' : 'backward_graph_dict_interpolate',
        'a' : {
        'class' : 'FusionRoot',
        'children' : ['stem:a','stem:b','stem:c','stem:d']
        }
    },
    'stem': {
        'a' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion', 'class:a', 'layer1:a', 'layer2:a']
            },
        'b' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'c' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'd' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'type': fusion_type,
            'in_channels': 64 * 4,
            'children': ['layer1:b','layer1:c','layer1:d']
            }
        },
    #in_channels: 64
    'layer1': {
        'a' : {
            'class': interpolation_type,
            'mode' : interpolation_mode,
            'input_channels' : 64,
            'stride' : 2,
            'children': ['layer1:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion', 'class:b', 'layer2:b']
            },
        'c' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'd' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 64 + 128 * 3,
            'type': fusion_type,
            'children': ['layer2:c', 'layer2:d']
            }
        },
    'layer2': {
        'a' : {
            'class': interpolation_type,
            'mode' : interpolation_mode,
            'input_channels' : 64,
            'stride' : 4,
            'children': ['layer2:fusion']
            },
        'b' : {
            'class': interpolation_type,
            'mode' : interpolation_mode,
            'input_channels' : 128,
            'stride' : 2,
            'children': ['layer2:fusion']
            },
        'c': {  # TODO add child 'class:c' and remove correspondingly below
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion', 'class:c']
            },
        'd' : { 
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 64 + 128 + 2 * 256,
            'type': fusion_type,
            'children': ['layer3:d']
            }
        },
    'layer3': {
        'd' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:d']
            },
        },
    'class': {
        'a' : {
            'class': 'FusionClassification',
            'input_channels' : 64,
            'classes' : 2,
            'children': ['output:a']
            },
        'b' : {
            'class': 'FusionClassification',
            'input_channels' : 128,
            'classes' : 4,
            'children': ['output:a']
            },
        'c' : {
            'class': 'FusionClassification',
            'input_channels' : 256,
            'classes' : 44,
            'children': ['output:a']
            },
        'd' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 197,
            'children': ['output:a']
            },
        },
    'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 4,
            'children' : []
            }
        }
    }
    return graph_dict

def make_backward_graph_dict_shared_stem(fusion_type):
    graph_dict = {
    'root': {
        'name' : 'backward_graph_dict_shared_stem',
        'a' : {
        'class' : 'FusionRootWithStem',
        'children' : ['stem:a','stem:b','stem:c','stem:d']
        }
    },
    'stem': {
        'a' : {
            'class': 'FusionFirstLayer',
            'blocks' : 1,
            'children': ['stem:fusion', 'class:a', 'layer1:a', 'layer2:a']
            },
        'b' : {
            'class': 'FusionFirstLayer',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'c' : {
            'class': 'FusionFirstLayer',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'd' : {
            'class': 'FusionFirstLayer',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 64 * 4,
            'type': fusion_type,
            'children': ['layer1:b','layer1:c','layer1:d']
            }
        },
    'layer1': {
        'a' : {
            'class': 'FusionDownsample',
            'input_channels' : 64,
            'stride' : 2,
            'children': ['layer1:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion', 'class:b', 'layer2:b']
            },
        'c' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'd' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 128 * 4,
            'type': fusion_type,
            'children': ['layer2:c', 'layer2:d']
            }
        },
    'layer2': {
        'a' : {
            'class': 'FusionDownsample',
            'input_channels' : 64,
            'stride' : 4,
            'children': ['layer2:fusion']
            },
        'b' : {
            'class': 'FusionDownsample',
            'input_channels' : 128,
            'stride' : 2,
            'children': ['layer2:fusion']
            },
        'c': {  # TODO add child 'class:c' and remove correspondingly below
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion', 'class:c']
            },
        'd' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 256 * 4,
            'type': fusion_type,
            'children': ['layer3:d']
            }
        },
    'layer3': {
        'd' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:d']
            },
        },
    'class': {
        'a' : {
            'class': 'FusionClassification',
            'input_channels' : 64,
            'classes' : 2,
            'children': ['output:a']
            },
        'b' : {
            'class': 'FusionClassification',
            'input_channels' : 128,
            'classes' : 4,
            'children': ['output:a']
            },
        'c' : {
            'class': 'FusionClassification',
            'input_channels' : 256,
            'classes' : 44,
            'children': ['output:a']
            },
        'd' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 197,
            'children': ['output:a']
            },
        },
    'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 4,
            'children' : []
            }
        }
    }
    return graph_dict

def make_forward_graph_dict(fusion_type):
    graph_dict = {
    'root': {
        'name' : 'forward_graph_dict',
        'a' : {
        'class' : 'FusionRoot',
        'children' : ['stem:a']
        }
    },
    'stem': {
        'a' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 64,
            'type': fusion_type,
            'children': ['layer1:a','layer1:b']
            }
        },
    'layer1': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 128 * 2,
            'type': fusion_type,
            'children': ['layer2:a', 'layer2:b', 'layer2:c']
            }
        },
    'layer2': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'c' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 256 * 3,
            'type': fusion_type,
            'children': ['layer3:a', 'layer3:b', 'layer3:c', 'layer3:d']
            }
        },
    'layer3': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:a']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:b']
            },
        'c' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:c']
            },
        'd' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:d']
            }
        },
    'class': {
        'd' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 2,
            'children': ['output:a']
            },
        'c' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 4,
            'children': ['output:a']
            },
        'b' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 44,
            'children': ['output:a']
            },
        'a' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 197,
            'children': ['output:a']
            },
        },
    'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 4,
            'children' : []
            }
        }
    }
    return graph_dict

def make_sideways_graph_dict(fusion_type):
    return {
        'root': {
            'name' : 'sideways_graph_dict',
            'a' : {
                'class' : 'FusionRoot',
                'children' : ['layer0:a', 'layer1:b', 'layer2:c', 'layer3:d']
            }
        },
        'layer0': {
            'a' : {
                'class' : 'FusionStemWithArgs',
                'blocks' : 1,
                'out_channels' : 64,
                'stride' : 2,
                'children' : ['layer1:a']
            },
        },
        'layer1': {
            'a' : {
                'class' : 'FusionLayer',
                'blocks': 1,
                'children': ['layer1:fusion']
            },
            'b' : {
                'class' : 'FusionStemWithArgs',
                'blocks' : 1,
                'out_channels' : 128,
                'stride' : 4,
                'children' : ['layer1:fusion']
            },
            'fusion' : {
                'class' : 'FusionMethod',
                'in_channels' : 128 * 2,
                'type': fusion_type,
                'children': ['layer2:a','layer2:b']
            }
        },
        'layer2' : {
            'a' : {
                'class' : 'FusionLayer',
                'blocks': 1,
                'children': ['layer2:fusion']
            },
            'b' : {
                'class' : 'FusionLayer',
                'blocks': 1,
                'children': ['layer2:fusion']
            },
            'c' : {
                'class' : 'FusionStemWithArgs',
                'blocks' : 1,
                'out_channels' : 256,
                'stride' : 8,
                'children' : ['layer2:fusion']
            },
            'fusion' : {
                'class' : 'FusionMethod',
                'in_channels' : 256 * 3,
                'type': fusion_type,
                'children': ['layer3:a','layer3:b', 'layer3:c']
            }
        },
        'layer3' : {
            'a' : {
                'class' : 'FusionLayer',
                'blocks': 1,
                'children': ['layer3:fusion']
            },
            'b' : {
                'class' : 'FusionLayer',
                'blocks': 1,
                'children': ['layer3:fusion']
            },
            'c' : {
                'class' : 'FusionLayer',
                'blocks': 1,
                'children': ['layer3:fusion']
            },
            'd' : {
                'class' : 'FusionStemWithArgs',
                'blocks' : 1,
                'out_channels' : 512,
                'stride' : 16,
                'children' : ['layer3:fusion']
            },
            'fusion' : {
                'class' : 'FusionMethod',
                'in_channels' : 512 * 4,
                'type': fusion_type,
                'children': ['class:a','class:b', 'class:c', 'class:d']
            }
        },
        'class': {
        'd' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 2,
            'children': ['output:a']
            },
        'c' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 4,
            'children': ['output:a']
            },
        'b' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 44,
            'children': ['output:a']
            },
        'a' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 197,
            'children': ['output:a']
            },
        },
        'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 4,
            'children' : []
            }
        }
    }

def make_forward_graph_dict_implicit_encoder_spatial(fusion_type):
    graph_dict = {
    'root': {
        'name' : 'forward_graph_dict_implicit_encoder_spatial',
        'a' : {
        'class' : 'FusionRoot',
        'children' : ['stem:a']
        }
    },
    'stem': {
        'a' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 64,
            'type': fusion_type,
            'children': ['layer1:a','layer1:b']
            }
        },
    'layer1': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 128 * 2,
            'type': fusion_type,
            'children': ['layer2:a', 'layer2:b', 'layer2:c']
            }
        },
    'layer2': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'c' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 256 * 3,
            'type': fusion_type,
            'children': ['layer3:a', 'layer3:b', 'layer3:c', 'layer3:d', 'layer3:reconstruction1']
            }
        },
    'layer3': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:a']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:b']
            },
        'c' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:c']
            },
        'd' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:d']
            },
        'reconstruction1' : {
            'class' : 'PoolEncoder',
            'output_size' : (7,14),
            'mode' : 'max',
            'children' : ['layer3:reconstruction2']
        }, 
        'reconstruction2' : {
            'class' : 'SpatialDecoder',
            'num_blocks' : 4,
            'start_channels' : 256,
            'upsample_with_interpolation' : True,
            'children' : ['output:a']
        }
        },
    'class': {
        'd' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 2,
            'children': ['output:a']
            },
        'c' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 4,
            'children': ['output:a']
            },
        'b' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 44,
            'children': ['output:a']
            },
        'a' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 197,
            'children': ['output:a']
            },
        },
    'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 5,
            'children' : []
            }
        }
    }
    return graph_dict

def make_forward_graph_dict_implicit_encoder_flat(fusion_type):
    graph_dict = {
    'root': {
        'name' : 'forward_graph_dict_implicit_encoder_flat',
        'a' : {
        'class' : 'FusionRoot',
        'children' : ['stem:a']
        }
    },
    'stem': {
        'a' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 64,
            'type': fusion_type,
            'children': ['layer1:a','layer1:b']
            }
        },
    'layer1': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 128 * 2,
            'type': fusion_type,
            'children': ['layer2:a', 'layer2:b', 'layer2:c']
            }
        },
    'layer2': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'c' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 256 * 3,
            'type': fusion_type,
            'children': ['layer3:a', 'layer3:b', 'layer3:c', 'layer3:d', 'layer3:recon']
            }
        },
    'layer3': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:a']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:b']
            },
        'c' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:c']
            },
        'd' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:d']
            },
        'recon' : {
            'class' : 'PoolEncoder',
            'output_size' : (7,14),
            'mode' : 'max',
            'children' : ['layer3:recon2']
        },
        'recon2' : {
            'class' : 'ConvPreprocess',
            'in_channels' : 256,
            'kernel_size' : (7,14),
            'children' : ['layer3:recon3']
        },
        'recon3' : {
            'class' : 'FlatDecoder',
            'num_blocks' : 4,
            'start_channels' : 256,
            'upsample_with_interpolation' : False,
            'children' : ['output:a']
        }
        },
    'class': {
        'd' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 2,
            'children': ['output:a']
            },
        'c' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 4,
            'children': ['output:a']
            },
        'b' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 44,
            'children': ['output:a']
            },
        'a' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 197,
            'children': ['output:a']
            },
        },
    'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 5,
            'children' : []
            }
        }
    }
    return graph_dict

def make_forward_graph_dict_bidirectional_decoder_spatial(fusion_type):
    graph_dict = {
        'root': {
            'name' : 'forward_graph_dict_bidirectional_decoder_spatial',
            'a': {
                'class': 'FusionRoot',
                'children': ['stem:a', 'stem:encoder_1']
            }
        },
        'stem': {
            'encoder_1': {
                'class': 'EncoderBlock',
                'in_channels': 3,
                'out_channels': 64,
                'stride': 4,
                'children': ['stem:encoder_2']  # hw = (56, 112)
            },
            'encoder_2': {
                'class': 'EncoderBlock',
                'in_channels': 64,
                'out_channels': 128,
                'stride': 2,
                # hw = (28, 56)
                'children': ['stem:encoder_3']
            },
            'encoder_3': {
                'class': 'EncoderBlock',
                'in_channels': 128,
                'out_channels': 256,
                'stride': 2,
                # hw = (14, 28)
                'children': ['stem:encoder_4']
            },
            'encoder_4': {
                'class': 'EncoderBlock',
                'in_channels': 256,
                'out_channels': 512,
                'stride': 2,
                'children': ['stem:decoder']  # hw = (7, 14)
            },
            'decoder': {
                'class': 'DecoderBlock',
                'in_channels': 512,
                'out_channels': 256,
                'stride': 2,
                # hw = (14, 28)
                'children': ['layer1:recon_fusion', 'stem:resample_1']
            },
            'resample_1': {
                'class': 'FusionResample',
                'scale_factor': 4.0,
                'in_channels': 256,
                'out_channels': 64,
                'children': ['stem:fusion']
            },
            'a': {
                'class': 'FusionStem',
                'blocks': 1,
                'children': ['stem:fusion']
            },
            'fusion': {
                'class': 'FusionMethod',
                'in_channels': 64*2,
                'out_channels': 64,
                'inputs': 2,
                'type': fusion_type,
                'children': ['layer1:a', 'layer1:b', 'stem:resample_2']
            },
            'resample_2' : {
                'class' : 'FusionResample',
                'scale_factor' : 1/4,
                'in_channels' : 64,
                'out_channels' : 256,
                'children' : ['layer1:recon_fusion']
            }
        },
        
        'layer1': {
            'recon_fusion' : {
                'class' : 'FusionMethod',
                'in_channels' : 256 * 2,
                'out_channels' : 256,
                'inputs' : 2, 
                'type': 'add',
                'children' : ['layer1:decoder']
            },
            'a': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer1:fusion']
            },
            'b': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer1:fusion']
            },
            'decoder': {
                'class': 'DecoderBlock',
                'in_channels': 256,
                'out_channels': 128,
                'stride': 2,
                # hw = (28, 56)
                'children': ['layer2:recon_fusion', 'layer1:resample_1']
            },
            'resample_1': {
                'class': 'FusionResample',
                'scale_factor': 1.0,
                'in_channels': 128,
                'out_channels': 128,
                'children': ['layer1:fusion']
            },
            'fusion': {
                'class': 'FusionMethod',
                'in_channels': 128 * 3,
                'out_channels': 128,
                'inputs': 3,
                'type': fusion_type,
                'children': ['layer2:a', 'layer2:b', 'layer1:resample_2']
            },
            'resample_2': {
                'class': 'FusionResample',
                'scale_factor': 1.0,
                'in_channels': 128,
                'out_channels': 128,
                'children': ['layer2:recon_fusion']
            }
        },
        'layer2': {
            'recon_fusion' : {
                'class' : 'FusionMethod',
                'in_channels' : 128 * 2,
                'out_channels' : 128,
                'inputs' : 2, 
                'type': 'add',
                'children' : ['layer2:decoder']
            },
            'a': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer2:fusion']
            },
            'b': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer2:fusion']
            },
            'decoder': {
                'class': 'DecoderBlock',
                'in_channels': 128,
                'out_channels': 64,
                'stride': 2,
                # hw (56, 112)
                'children': ['layer2:resample_1', 'layer3:recon_fusion']
            },
            'resample_1': {
                'class': 'FusionResample',
                'scale_factor': 0.25,
                'in_channels': 64,
                'out_channels': 256,
                'children' : ['layer2:fusion']
            },
            'fusion': {
                'class': 'FusionMethod',
                'in_channels': 256 * 3,
                'out_channels': 256,
                'inputs': 3,
                'type': fusion_type,
                'children': ['layer3:a', 'layer3:b', 'layer2:resample_2']
            },
            'resample_2': {
                'class': 'FusionResample',
                'scale_factor': 4.0,
                'in_channels': 256,
                'out_channels': 64,
                'children' : ['layer3:recon_fusion']
            },
        },
        'layer3': {
            'recon_fusion' : {
                'class' : 'FusionMethod',
                'in_channels' : 64 * 2,
                'out_channels' : 64,
                'inputs' : 2, 
                'type': 'add',
                'children' : ['layer3:decoder']
            },
            'decoder': {  # (112, 224)
                'class': 'DecoderBlock',
                'in_channels': 64,
                'out_channels': 32,
                'stride': 2,
                'children': ['class:decoder']
            },
            'a': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['class:a']
            },
            'b': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['class:b']
            },
        },
        'class': {
            'decoder': {
                'class': 'PostDecoderBlock',
                'in_channels': 32,
                'children': ['output:a']
            },
            'b': {
                'class': 'FusionClassification',
                'input_channels': 512,
                'classes': 44,
                'children': ['output:a']
            },
            'a': {
                'class': 'FusionClassification',
                'input_channels': 512,
                'classes': 197,
                'children': ['output:a']
            }
        },
        'output': {
            'a': {
                'class': 'FusionOutput',
                'inputs': 3,
                'children': []
            }
        }
    }
    return graph_dict

def make_forward_graph_dict_bidirectional_decoder_flat(fusion_type, decoder_fusion_type):
    graph_dict = {
        'root': {
            'name' : 'forward_graph_dict_bidirectional_decoder_flat',
            'a': {
                'class': 'FusionRoot',
                'children': ['stem:a', 'stem:encoder_1']
            }
        },
        'stem': {
            'encoder_1': {
                'class': 'EncoderBlock',
                'in_channels': 3,
                'out_channels': 64,
                'stride': 4,
                'children': ['stem:encoder_2']  # hw = (56, 112)
            },
            'encoder_2': {
                'class': 'EncoderBlock',
                'in_channels': 64,
                'out_channels': 128,
                'stride': 2,
                # hw = (28, 56)
                'children': ['stem:encoder_3']
            },
            'encoder_3': {
                'class': 'EncoderBlock',
                'in_channels': 128,
                'out_channels': 256,
                'stride': 2,
                # hw = (14, 28)
                'children': ['stem:encoder_4']
            },
            'encoder_4': {
                'class': 'EncoderBlock',
                'in_channels': 256,
                'out_channels': 512,
                'stride': 2,
                'children': ['stem:post_encoder']  # hw = (7, 14)
            },
            'post_encoder' : {
                'class' : 'ConvPreprocess',
                'in_channels' : 512,
                'kernel_size' : (7,14),
                'children' : ['stem:pre_decoder']
            },
            'pre_decoder': {
                # start_channels, kernel_size, start_out_channels, upsample_with_interpolation
                'class': 'PreDecoderBlock',
                'start_channels': 512,
                'start_out_channels': 512,
                'kernel_size': (7,14),
                'children': ['stem:decoder']
            },
            'decoder': {
                'class': 'DecoderBlock',
                'in_channels': 512,
                'out_channels': 256,
                'stride': 2,
                # hw = (14, 28)
                'children': ['layer1:recon_fusion', 'stem:resample_1']
            },
            'resample_1': {
                'class': 'FusionResample',
                'scale_factor': 4.0,
                'in_channels': 256,
                'out_channels': 64,
                'children': ['stem:fusion']
            },
            'a': {
                'class': 'FusionStem',
                'blocks': 1,
                'children': ['stem:fusion']
            },
            'fusion': {
                'class': 'FusionMethod',
                'in_channels': 64*2,
                'out_channels': 64,
                'inputs': 2,
                'type': fusion_type,
                'children': ['layer1:a', 'layer1:b', 'stem:resample_2']
            },
            'resample_2' : {
                'class' : 'FusionResample',
                'scale_factor' : 1/4,
                'in_channels' : 64,
                'out_channels' : 256,
                'children' : ['layer1:recon_fusion']
            }
        },
        
        'layer1': {
            'recon_fusion' : {
                'class' : 'FusionMethod',
                'in_channels' : 256 * 2,
                'out_channels' : 256,
                'inputs' : 2, 
                'type': decoder_fusion_type,
                'children' : ['layer1:decoder']
            },
            'a': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer1:fusion']
            },
            'b': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer1:fusion']
            },
            'decoder': {
                'class': 'DecoderBlock',
                'in_channels': 256,
                'out_channels': 128,
                'stride': 2,
                # hw = (28, 56)
                'children': ['layer2:recon_fusion', 'layer1:resample_1']
            },
            'resample_1': {
                'class': 'FusionResample',
                'scale_factor': 1.0,
                'in_channels': 128,
                'out_channels': 128,
                'children': ['layer1:fusion']
            },
            'fusion': {
                'class': 'FusionMethod',
                'in_channels': 128 * 3,
                'out_channels': 128,
                'inputs': 3,
                'type': fusion_type,
                'children': ['layer2:a', 'layer2:b', 'layer1:resample_2']
            },
            'resample_2': {
                'class': 'FusionResample',
                'scale_factor': 1.0,
                'in_channels': 128,
                'out_channels': 128,
                'children': ['layer2:recon_fusion']
            }
        },
        'layer2': {
            'recon_fusion' : {
                'class' : 'FusionMethod',
                'in_channels' : 128 * 2,
                'out_channels' : 128,
                'inputs' : 2, 
                'type': decoder_fusion_type,
                'children' : ['layer2:decoder']
            },
            'a': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer2:fusion']
            },
            'b': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer2:fusion']
            },
            'decoder': {
                'class': 'DecoderBlock',
                'in_channels': 128,
                'out_channels': 64,
                'stride': 2,
                # hw (56, 112)
                'children': ['layer2:resample_1', 'layer3:recon_fusion']
            },
            'resample_1': {
                'class': 'FusionResample',
                'scale_factor': 0.25,
                'in_channels': 64,
                'out_channels': 256,
                'children' : ['layer2:fusion']
            },
            'fusion': {
                'class': 'FusionMethod',
                'in_channels': 256 * 3,
                'out_channels': 256,
                'inputs': 3,
                'type': fusion_type,
                'children': ['layer3:a', 'layer3:b', 'layer2:resample_2']
            },
            'resample_2': {
                'class': 'FusionResample',
                'scale_factor': 4.0,
                'in_channels': 256,
                'out_channels': 64,
                'children' : ['layer3:recon_fusion']
            },
        },
        'layer3': {
            'recon_fusion' : {
                'class' : 'FusionMethod',
                'in_channels' : 64 * 2,
                'out_channels' : 64,
                'inputs' : 2, 
                'type': decoder_fusion_type,
                'children' : ['layer3:decoder']
            },
            'decoder': {  # (112, 224)
                'class': 'DecoderBlock',
                'in_channels': 64,
                'out_channels': 32,
                'stride': 2,
                'children': ['class:decoder']
            },
            'a': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['class:a']
            },
            'b': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['class:b']
            },
        },
        'class': {
            'decoder': {
                'class': 'PostDecoderBlock',
                'in_channels': 32,
                'children': ['output:a']
            },
            'b': {
                'class': 'FusionClassification',
                'input_channels': 512,
                'classes': 44,
                'children': ['output:a']
            },
            'a': {
                'class': 'FusionClassification',
                'input_channels': 512,
                'classes': 197,
                'children': ['output:a']
            }
        },
        'output': {
            'a': {
                'class': 'FusionOutput',
                'inputs': 3,
                'children': []
            }
        }
    }
    return graph_dict


def make_forward_graph_dict_bidirectional_decoder_flat_dropout(fusion_type, decoder_fusion_type):
    graph_dict = {
        'root': {
            'name' : 'forward_graph_dict_bidirectional_decoder_flat_dropout',
            'a': {
                'class': 'FusionRoot',
                'children': ['stem:a', 'stem:encoder_1']
            }
        },
        'stem': {
            'encoder_1': {
                'class': 'EncoderBlock',
                'in_channels': 3,
                'out_channels': 64,
                'stride': 4,
                'children': ['stem:encoder_2']  # hw = (56, 112)
            },
            'encoder_2': {
                'class': 'EncoderBlock',
                'in_channels': 64,
                'out_channels': 128,
                'stride': 2,
                # hw = (28, 56)
                'children': ['stem:encoder_3']
            },
            'encoder_3': {
                'class': 'EncoderBlock',
                'in_channels': 128,
                'out_channels': 256,
                'stride': 2,
                # hw = (14, 28)
                'children': ['stem:encoder_4']
            },
            'encoder_4': {
                'class': 'EncoderBlock',
                'in_channels': 256,
                'out_channels': 512,
                'stride': 2,
                'children': ['stem:post_encoder']  # hw = (7, 14)
            },
            'post_encoder': {
                'class': 'ConvPreprocess',
                'in_channels': 512,
                'kernel_size': (7, 14),
                'children': ['stem:pre_decoder']
            },
            'pre_decoder': {
                # start_channels, kernel_size, start_out_channels, upsample_with_interpolation
                'class': 'PreDecoderBlock',
                'start_channels': 512,
                'start_out_channels': 512,
                'kernel_size': (7, 14),
                'children': ['stem:decoder']
            },
            'decoder': {
                'class': 'DecoderBlock',
                'in_channels': 512,
                'out_channels': 256,
                'stride': 2,
                # hw = (14, 28)
                'children': ['layer1:recon_fusion', 'stem:resample_1']
            },
            'resample_1': {
                'class': 'FusionResample',
                'scale_factor': 4.0,
                'in_channels': 256,
                'out_channels': 64,
                'children': ['stem:fusion']
            },
            'a': {
                'class': 'FusionStem',
                'blocks': 1,
                'children': ['stem:fusion']
            },
            'fusion': {
                'class': 'FusionMethod',
                'in_channels': 64*2,
                'out_channels': 64,
                'inputs': 2,
                'type': fusion_type,
                'children': ['layer1:a', 'layer1:b', 'stem:resample_2']
            },
            'resample_2': {
                'class': 'FusionResample',
                'scale_factor': 1/4,
                'in_channels': 64,
                'out_channels': 256,
                'children': ['stem:dropout']
            },
            'dropout':{
                'class' : 'DropoutBlock',
                'probability': 0.5,
                'dims': 1,
                'children' : ['layer1:recon_fusion']
            }
        },

        'layer1': {
            'recon_fusion': {
                'class': 'FusionMethod',
                'in_channels': 256 * 2,
                'out_channels': 256,
                'inputs': 2,
                'type': decoder_fusion_type,
                'children': ['layer1:decoder']
            },
            'a': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer1:fusion']
            },
            'b': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer1:fusion']
            },
            'decoder': {
                'class': 'DecoderBlock',
                'in_channels': 256,
                'out_channels': 128,
                'stride': 2,
                # hw = (28, 56)
                'children': ['layer2:recon_fusion', 'layer1:resample_1']
            },
            'resample_1': {
                'class': 'FusionResample',
                'scale_factor': 1.0,
                'in_channels': 128,
                'out_channels': 128,
                'children': ['layer1:fusion']
            },
            'fusion': {
                'class': 'FusionMethod',
                'in_channels': 128 * 3,
                'out_channels': 128,
                'inputs': 3,
                'type': fusion_type,
                'children': ['layer2:a', 'layer2:b', 'layer1:resample_2']
            },
            'resample_2': {
                'class': 'FusionResample',
                'scale_factor': 1.0,
                'in_channels': 128,
                'out_channels': 128,
                'children': ['layer1:dropout']
            },
            'dropout':{
                'class' : 'DropoutBlock',
                'probability': 0.75,
                'dims': 1,
                'children' : ['layer2:recon_fusion']
            }
        },
        'layer2': {
            'recon_fusion': {
                'class': 'FusionMethod',
                'in_channels': 128 * 2,
                'out_channels': 128,
                'inputs': 2,
                'type': decoder_fusion_type,
                'children': ['layer2:decoder']
            },
            'a': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer2:fusion']
            },
            'b': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['layer2:fusion']
            },
            'decoder': {
                'class': 'DecoderBlock',
                'in_channels': 128,
                'out_channels': 64,
                'stride': 2,
                # hw (56, 112)
                'children': ['layer2:resample_1', 'layer3:recon_fusion']
            },
            'resample_1': {
                'class': 'FusionResample',
                'scale_factor': 0.25,
                'in_channels': 64,
                'out_channels': 256,
                'children': ['layer2:fusion']
            },
            'fusion': {
                'class': 'FusionMethod',
                'in_channels': 256 * 3,
                'out_channels': 256,
                'inputs': 3,
                'type': fusion_type,
                'children': ['layer3:a', 'layer3:b', 'layer2:resample_2']
            },
            'resample_2': {
                'class': 'FusionResample',
                'scale_factor': 4.0,
                'in_channels': 256,
                'out_channels': 64,
                'children': ['layer2:dropout']
            },
            'dropout':{
                'class' : 'DropoutBlock',
                'probability': 0.25,
                'dims': 2,
                'children' : ['layer3:recon_fusion']
            }
        },
        'layer3': {
            'recon_fusion': {
                'class': 'FusionMethod',
                'in_channels': 64 * 2,
                'out_channels': 64,
                'inputs': 2,
                'type': decoder_fusion_type,
                'children': ['layer3:decoder']
            },
            'decoder': {  # (112, 224)
                'class': 'DecoderBlock',
                'in_channels': 64,
                'out_channels': 32,
                'stride': 2,
                'children': ['class:decoder']
            },
            'a': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['class:a']
            },
            'b': {
                'class': 'FusionLayer',
                'blocks': 1,
                'children': ['class:b']
            },
        },
        'class': {
            'decoder': {
                'class': 'PostDecoderBlock',
                'in_channels': 32,
                'children': ['output:a']
            },
            'b': {
                'class': 'FusionClassification',
                'input_channels': 512,
                'classes': 44,
                'children': ['output:a']
            },
            'a': {
                'class': 'FusionClassification',
                'input_channels': 512,
                'classes': 197,
                'children': ['output:a']
            }
        },
        'output': {
            'a': {
                'class': 'FusionOutput',
                'inputs': 3,
                'children': []
            }
        }
    }
    return graph_dict

def make_forward_graph_dict_encoder_decoder_feedback(fusion_type):
    graph_dict = {
    'root': {
        'name' : 'forward_graph_dict_encoder_decoder_feedback',
        'a' : {
        'class' : 'FusionRoot',
        'children' : ['stem:a', 'stem:encoder_1']
        }
    },
    'stem': {
        'encoder_1': {
            'class' : 'EncoderBlock',
            'in_channels' : 3,
            'out_channels' : 64,
            'stride' : 4,
            'children': ['stem:fusion', 'stem:encoder_2'] # hw = (56, 112)
        },
        'encoder_2': {
            'class' : 'EncoderBlock',
            'in_channels' : 64,
            'out_channels' : 128,
            'stride' : 2,
            'children': ['layer1:fusion', 'stem:encoder_3'] #hw = (28, 56)
        },
        'encoder_3': {
            'class' : 'EncoderBlock',
            'in_channels' : 128,
            'out_channels' : 256,
            'stride' : 2,
            'children': ['layer2:fusion', 'stem:encoder_4'] #hw = (14, 28)
        },
        'encoder_4': {
            'class' : 'EncoderBlock',
            'in_channels' : 256,
            'out_channels' : 512,
            'stride' : 2,
            'children': ['stem:decoder_1'] #hw = (7, 14)
        },
        'decoder_1':{
            'class' : 'DecoderBlock',
            'in_channels' : 512,
            'out_channels' : 256,
            'stride' : 2,
            'children' : ['stem:decoder_2', 'stem:resample_1'] #hw = (14, 28)
        },
        'resample_1':{
            'class' : 'FusionResample',
            'scale_factor' : 4.0,
            'in_channels' : 256,
            'out_channels' : 64,
            'children' : ['stem:fusion']
        },
        'decoder_2':{
            'class' : 'DecoderBlock',
            'in_channels' : 256,
            'out_channels' : 128,
            'stride' : 2,
            'children' : ['stem:decoder_3', 'stem:resample_2'] #hw = (28, 56)
        },
        'resample_2':{
            'class' : 'FusionResample',
            'scale_factor' : 1.0,
            'in_channels' : 128,
            'out_channels' : 128,
            'children' : ['layer1:fusion']
        },
        'decoder_3':{
            'class' : 'DecoderBlock',
            'in_channels' : 128,
            'out_channels' : 64,
            'stride' : 2,
            'children' : ['stem:decoder_4', 'stem:resample_3'] #hw (56, 112)
        },
        'resample_3':{
            'class' : 'FusionResample',
            'scale_factor' : 0.25,
            'in_channels' : 64,
            'out_channels' : 256,
            'children' : ['layer2:fusion']
        },
        'decoder_4':{ # (112, 224)
            'class' : 'DecoderBlock',
            'in_channels' : 64,
            'out_channels' : 32,
            'stride' : 2,
            'children' : ['stem:decoder_5']
        },
        'decoder_5':{
            'class': 'PostDecoderBlock',
            'in_channels': 32,
            'children': ['output:a']
        },
        'a' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 64*3,
            'out_channels' : 64,
            'inputs' : 3,
            'type': fusion_type,
            'children': ['layer1:a','layer1:b']
            }
        },
    'layer1': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 128 * 4,
            'out_channels' : 128,
            'inputs': 4,
            'type': fusion_type,
            'children': ['layer2:a', 'layer2:b']
            }
        },
    'layer2': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 256 * 4,
            'out_channels' : 256,
            'inputs': 4,
            'type': fusion_type,
            'children': ['layer3:a', 'layer3:b']
            }
        },
    'layer3': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:a']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:b']
            },
        },
    'class': {
        'b' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 44,
            'children': ['output:a']
            },
        'a' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 197,
            'children': ['output:a']
            }
        },
    'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 3,
            'children' : []
            }
        }
    }
    return graph_dict


def make_forward_graph_dict_explicit_encoder_spatial(fusion_type, use_interpolation):
    graph_dict = {
    'root': {
        'name' : 'forward_graph_dict_explicit_encoder_spatial',
        'a' : {
        'class' : 'FusionRoot',
        'children' : ['stem:a']
        }
    },
    'stem': {
        'a' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 64,
            'type': fusion_type,
            'children': ['layer1:a','layer1:b','layer1:recon']
            }
        },
    'layer1': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'recon': {
            'class': 'EncoderBlock',
            'in_channels': 64,
            'out_channels': 128,
            'pool_mode' : None,
            'children':['layer1:fusion', 'layer2:recon']
        },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 128 * 3,
            'inputs' : 3,
            'out_channels' : 128,
            'type': fusion_type,
            'children': ['layer2:a', 'layer2:b']
            }
        },
    'layer2': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'recon': {
            'class': 'EncoderBlock',
            'in_channels': 128,
            'out_channels' : 256,
            'pool_mode' : None,
            'children':['layer2:fusion', 'layer3:recon']
        },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 256 * 3,
            'inputs' : 3,
            'out_channels' : 256,
            'type': fusion_type,
            'children': ['layer3:a', 'layer3:b']
            }
        },
    'layer3': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:a']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:b']
            },
        'recon' : {
            'class' : 'EncoderBlock',
            'in_channels' : 256,
            'out_channels' : 512,
            'pool_mode' : None,
            'children' : ['class:recon2']
        }
        },
    'class': {
        'recon2' : {
            'class' : 'SpatialDecoder',
            'num_blocks' : 4,
            'start_channels' : 512,
            'upsample_with_interpolation' : use_interpolation,
            'children' : ['output:a']
        },
        'b' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 44,
            'children': ['output:a']
            },
        'a' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 197,
            'children': ['output:a']
            }
        },
    'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 3,
            'children' : []
            }
        }
    }
    return graph_dict

def make_forward_graph_dict_explicit_encoder_flat(fusion_type, preprocess_type):
    graph_dict = {
    'root': {
        'name' : 'forward_graph_dict_explicit_encoder_flat',
        'a' : {
        'class' : 'FusionRoot',
        'children' : ['stem:a']
        }
    },
    'stem': {
        'a' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:fusion']
            },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 64,
            'type': fusion_type,
            'children': ['layer1:a','layer1:b','layer1:recon']
            }
        },
    'layer1': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer1:fusion']
            },
        'recon': {
            'class': 'EncoderBlock',
            'in_channels': 64,
            'out_channels' : 128,
            'pool_mode' : None,
            'children':['layer1:fusion', 'layer2:recon']
        },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 128 * 3,
            'out_channels' : 128,
            'inputs' : 3,
            'type': fusion_type,
            'children': ['layer2:a', 'layer2:b']
            }
        },
    'layer2': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['layer2:fusion']
            },
        'recon': {
            'class': 'EncoderBlock',
            'in_channels': 128,
            'out_channels' : 256,
            'pool_mode' : None,
            'children':['layer2:fusion', 'layer3:recon']
        },
        'fusion': {
            'class': 'FusionMethod',
            'in_channels' : 256 * 3,
            'out_channels' : 256,
            'inputs' : 3,
            'type': fusion_type,
            'children': ['layer3:a', 'layer3:b']
            }
        },
    'layer3': {
        'a' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:a']
            },
        'b' : {
            'class': 'FusionLayer',
            'blocks' : 1,
            'children': ['class:b']
            },
        'recon' : {
            'class' : 'EncoderBlock',
            'in_channels' : 256,
            'out_channels' : 512,
            'pool_mode' : None,
            'children' : ['class:recon']
        }
        },
    'class': {
        'recon' : {
            'class' : preprocess_type,
            'in_channels' : 512,
            'kernel_size' : (7,14),
            'start_out_channels' : 512,
            'children' : ['class:recon2']
        },
        'recon2' : {
            'class' : 'FlatDecoder',
            'num_blocks' : 4,
            'start_channels' : 512,
            'upsample_with_interpolation' : False,
            'children' : ['output:a']
        },
        'b' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 44,
            'children': ['output:a']
            },
        'a' : {
            'class': 'FusionClassification',
            'input_channels' : 512,
            'classes' : 197,
            'children': ['output:a']
            }
        },
    'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 3,
            'children' : []
            }
        }
    }
    return graph_dict

def make_auto_encoder_flat(preprocess_type):
    graph_dict = {
    'root': {
        'name' : 'make_auto_encoder_flat',
        'a' : {
        'class' : 'FusionRoot',
        'children' : ['stem:a']
        }
    },
    'stem': {
        'a' : {
            'class': 'FusionStem',
            'blocks' : 1,
            'children': ['stem:recon']
        },
        'recon': {
            'class': 'EncoderBlock',
            'in_channels': 64,
            'out_channels' : 128,
            'pool_mode' : None,
            'children':['stem:recon1']
        },
        'recon1': {
            'class': 'EncoderBlock',
            'in_channels': 128,
            'out_channels' : 256,
            'pool_mode' : None,
            'children':['stem:recon2']
        },
        'recon2' : {
            'class' : 'EncoderBlock',
            'in_channels' : 256,
            'out_channels' : 512,
            'pool_mode' : None,
            'children' : ['stem:recon3']
        },
        'recon3' : {
            'class' : preprocess_type,
            'in_channels' : 512,
            'kernel_size' : (7,14),
            'start_out_channels' : 512,
            'children' : ['class:recon']
        }
    },
    'class': {
        'recon' : {
            'class' : 'FlatDecoder',
            'num_blocks' : 4,
            'start_channels' : 512,
            'upsample_with_interpolation' : False,
            'children' : ['output:a']
        }
    },
    'output': {
        'a' : {
            'class': 'FusionOutput',
            'inputs' : 1,
            'children' : []
            }
        }
    }
    return graph_dict