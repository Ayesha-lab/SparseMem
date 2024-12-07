import os
import time
import random
from copy import deepcopy
import numpy as np
from ayeshas_code.utils import append_level_to_json, get_neurons_to_prune
from ayeshas_code.pruners.prunitmem2 import get_unitmem
from ayeshas_code.CustomPruningMethod import IndexedPruningMethod
import torch.nn.utils.prune as prune
import torch.nn as nn


def get_l1_scores(model):
    l1_score_dict = {}
    conv_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d) and name != "conv1"]
    
    for name, conv_layer in conv_layers:
        start_time = time.time()
        weights = conv_layer.weight.data
        l1_scores = (weights.abs().sum(dim=(1,2,3))).tolist()

        l1_neuron_dict = {}
        for i, x in enumerate(l1_scores):
            l1_neuron_dict[i] = x

        l1_score_dict[name] = l1_neuron_dict
        
        current_time = time.time()
        process_time = current_time - start_time
        print(f"Processed layer: {name}, time: {process_time:.2f}s", flush=True)
 
    return l1_score_dict

def prune_by_lowest_l1(model, level, sparsity_list, checkpoint, result_path, args, prev_pruned_ind= None):
    
    # Load last model
    model.to('cuda')
    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    l1_structured_pruner = IndexedPruningMethod()
    
    # Pass the subset for getting unit_mem
    unit_mems = get_unitmem(model, level, result_path, args)
    print(f"Obtaining l1 scores of neurons for level {level}")
    l1_scores = get_l1_scores(model)
    
    sparsity = sparsity_list[level]
    indices_to_prune, updated_prev_pruned_ind = get_neurons_to_prune(l1_scores, level, args.x, 0 , args, prev_pruned_ind)
    
    total_ind_pruned = sum(len(value) for value in indices_to_prune.values())
    print(f'{total_ind_pruned} neurons will be pruned, at indices: \n {indices_to_prune}')
    
    # Save info
    pruned_ind_path = os.path.join(result_path, "pruned_indices.json")
    append_level_to_json(level, indices_to_prune, pruned_ind_path)
    l1_scores_path = os.path.join(result_path, "l1_scores.json")
    append_level_to_json(level, l1_scores, l1_scores_path)
    
    print(f"pruning lowest {args.x} from unpruned neurons, sparsity: {sparsity}%")
    print(f"hooks after level {level}")
    for layer_name, module in model.named_modules():
            if layer_name in indices_to_prune.keys():
                ind = indices_to_prune[layer_name]
                l1_structured_pruner.indices = ind
                l1_structured_pruner.apply(module, name="weight", indices = ind)
                
                
                for hook in module._forward_pre_hooks.values():
                    if hook._tensor_name == "weight":  
                        print(f"hook for {layer_name}:{hook}")
                        break
    
    
    return updated_prev_pruned_ind