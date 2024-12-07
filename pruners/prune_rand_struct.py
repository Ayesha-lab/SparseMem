import numpy as np
from copy import deepcopy
import random
from ayeshas_code.utils import append_level_to_json
from ayeshas_code.pruners.prunitmem2 import get_unitmem
import torch.nn.utils.prune as prune
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def flatten_dict(dictionary):
    flat_dict = []
    for layer, neuron_list in dictionary.items():
        flat_dict.extend((layer, neuron) for neuron in neuron_list)
        
    return flat_dict


def prune_random_neurons(model, level, sparsity_list, checkpoint, result_path, args, prev_pruned_ind= None):
    X = args.x 
    sparsity = sparsity_list[level]
    model.to('cuda')
    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Pass the subset for getting unit_mem
    unit_mems = get_unitmem(model, level, result_path, args)
    print(f"pruning random {X} from unpruned neurons, sparsity: {sparsity}%")
    # Create a dummy dictionary containing all neurons
    layer_neuron_idx = {layer: list(range(len(value))) for layer, value in unit_mems.items()}
    
    # If nothing was pruned, create an empty prev_pruned dict
    if prev_pruned_ind is None:
        prev_pruned_ind = {layer:[] for layer in unit_mems.keys()}
        total_neurons = sum(len(unitmems) for unitmems in unit_mems.values())
        total_to_prune = int(np.ceil(total_neurons*(sparsity_list[level]/100)))
    
    # Else take prev_pruned dict and remove pruned neurons from layer_neuron_idx      
    else:
        for layer, neuron in layer_neuron_idx.items():
            ind = prev_pruned_ind[layer]
            for i in sorted(ind, reverse=True):
                layer_neuron_idx[layer].pop(i) 
                    
        total_neurons = sum(len(neurons) for neurons in layer_neuron_idx.values())
        total_to_prune = int(np.ceil(total_neurons*(X/100))) 
        
    # Flatten and obtain neurons to prune based on sparsity    
    flat_dict_to_list = flatten_dict(layer_neuron_idx)
    shuffle_list = deepcopy(flat_dict_to_list)
    random.shuffle(shuffle_list)
    target_neurons = shuffle_list[:total_to_prune]
    
    # Update prev_pruned dict and create a dictionary of neurons to prune
    new_indices = {layer: [] for layer in unit_mems.keys()} 
    for (layer, neuron) in target_neurons:
        new_indices[layer].append(neuron)
        prev_pruned_ind[layer].append(neuron)  
        
    pruned_ind_path = os.path.join(result_path, "pruned_indices.json")        
    append_level_to_json(level, new_indices, pruned_ind_path) 
    
    # Apply mask
    for layer_name, module in model.named_modules():
        if layer_name in new_indices.keys():
            num_neurons = len(new_indices[layer_name])
            _ = prune.random_structured(module, "weight", num_neurons, dim = 0)
            
            
            for hook in module._forward_pre_hooks.values():
                if hook._tensor_name == "weight":  
                    print(f"hook for {layer_name}:{hook}")
                    break
    
    return prev_pruned_ind