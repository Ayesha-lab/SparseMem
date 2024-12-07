
import os 
import json
import csv
import ast
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from collections import OrderedDict
from torch.utils.data import DataLoader

def write_to_file(sublist, level, path):
    with open((path), 'a') as f:
        f.write(f"Level: {level}\n")
        f.write(f"Sublist: {sublist.tolist()}\n\n") 
        
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj
        
def append_level_to_json(level, dictionary, path):
    # Ensure the directory exists
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    # Read the existing data
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # Convert all non-serializable objects in the dictionary
    serializable_dictionary = convert_to_serializable(dictionary)

    # Add or update the level key with the new dictionary
    data[str(level)] = serializable_dictionary  # Convert level to string to ensure it's a valid JSON key

    # Write the updated data back to the file
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def create_experiment_folder(base_path): # goes to "/home/ayesha/SparseMem/ayeshas_code/results/unitmem"
    """Based on pruner type, creates an exp folder"""
    # Define the experiment prefix and format
    exp_prefix = "exp_"
    exp_dirs = [d for d in os.listdir(base_path) if d.startswith(exp_prefix) and os.path.isdir(os.path.join(base_path, d))]

    # Find the highest experiment number
    if exp_dirs:
        max_exp_number = max(int(d[len(exp_prefix)+1:]) for d in exp_dirs)
        new_exp_number = max_exp_number + 1
    else:
        new_exp_number = 0

    # Create the new experiment directory
    new_exp_folder = os.path.join(base_path, f"{exp_prefix}_{new_exp_number}") # "/home/ayesha/SparseMem/ayeshas_code/results/unitmem/exp_1"
    os.makedirs(new_exp_folder, exist_ok=True)


    # Create 'ckpts' subdirectory under the method_percent folder
    ckpts_folder = os.path.join(new_exp_folder, 'ckpts') # "/home/ayesha/SparseMem/ayeshas_code/results/unitmem/exp_1/ckpts"
    os.makedirs(ckpts_folder, exist_ok=True)

    final_path = new_exp_folder 
    # print(f"Created directories:\n{new_exp_folder}\n{ckpts_folder}, returning {final_path}")
    return final_path

def load_sublist(file_path, level_number):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        current_level = None
        for row in reader:
            if row:
                if row[0].startswith('Level'):
                    current_level = int(row[0].split(':')[1].strip())
                elif row[0].startswith('Sublist') and current_level == level_number:
                    sublist_str = row[0].split(':', 1)[1].strip()
                    try:
                        sublist = ast.literal_eval(sublist_str)
                        return np.array(sublist)
                    except (SyntaxError, ValueError) as e:
                        print(f"Error parsing sublist: {e}")
                        return None
    return None


   # Create and save subindices for computing unitmems on both methods
    # sublists = {}
    # for l in range(len(density_list[1:])):
    #     idx_sublist = np.random.choice(len(train_set), 500)
    #     write_to_file(idx_sublist, l, os.path.join(main_dir,f"indices_{exp_no}.csv"))
    #     sublists[l] = idx_sublist
    
def write_dict_to_json(dictionary, filename):
    with open(filename, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)
    
def read_file_to_dict(file_path):
    # Initialize an empty dictionary
    level_dict = {}

    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into lines
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    # Process the lines
    for i in range(0, len(lines), 2):
        if lines[i].startswith('Level:'):
            level = int(lines[i].split(': ')[1])
            sublist_str = lines[i+1].split(': ')[1]
            sublist = ast.literal_eval(sublist_str)
            level_dict[level] = sublist

    return level_dict

def load_pruned_indices_as_dict(exp_no):
    """Load the indices of the neurons pruned with UnitMem for this experiment"""
    base_dir = '/home/ayesha/SparseMem/ayeshas_code/results/unitmem'
    exp_dir = os.path.join(base_dir, f'exp__{exp_no}')
    target_file = 'pruned_indices.json'
    file_path = os.path.join(exp_dir, target_file)
    
    # if os.path.isfile(file_path):
        # Load the JSON file as an OrderedDict
    with open(file_path, 'r') as f:
        data = json.load(f) 
    return data
    # else:
    #     return None 


def flatten_to_tuple(unit_mem_dict):
    flatted_unitmems  = []
    for layer, um_values in unit_mem_dict.items():
        for idx, um in um_values.items():
            flatted_unitmems.append((layer,idx,um))
    
    return(flatted_unitmems)

def get_prev_percentage(X,level): 
    reduction_factor = 1 - (X / 100)
    prev_perc = 100 - round(100*(reduction_factor**(level)), 2)
    return prev_perc

def get_neurons_to_prune(unit_mems, level, lowest, highest, args, prev_pruned_ind = None):
    # update the unitmems to remove previously pruned indices
    updated_unitmems = copy.deepcopy(unit_mems)
    
    if prev_pruned_ind is None:
        prev_pruned_ind = {layer : [] for layer, _ in unit_mems.items()}   

    # remove previously pruned neurons
    for layer, UM_values in unit_mems.items():
        layer_in_prev_ind = layer in prev_pruned_ind.keys()
        is_not_first_conv = layer != "conv"
        if layer_in_prev_ind and is_not_first_conv:
            # get prev pruned indices for layer
            prev_ind = prev_pruned_ind[layer]
            # remove them from this layer in unitmems
            for ind in sorted(prev_ind, reverse=True):
                updated_unitmems[layer].pop(ind)
    
    acts = flatten_to_tuple(updated_unitmems)
    acts.sort(key=lambda x: x[2]) 
    
    prune_low_count = int(np.ceil(len(acts) * lowest / 100))
    prune_high_count = int(np.ceil(len(acts) * highest / 100)) # gets value to prune
    
    target_neurons = acts[:prune_low_count] # enter indices until a certain value
    if prune_high_count != 0:
        target_neurons = target_neurons + acts[-prune_high_count:] # if hybrid
        
    # Group activations by layer, maintaining input order
    new_indices = {layer : [] for layer, _ in unit_mems.items()}
    for layer, idx, activations in target_neurons:
        
        new_indices[layer].append(idx)
        prev_pruned_ind[layer].append(idx)

    return new_indices, prev_pruned_ind


def num_weights_to_prunev2(layer_pruned_neurons, model):
    
    weights_per_level = {}
    module_dict =  dict(model.named_modules())
    for level, layer_info in layer_pruned_neurons.items():
        # For this level, get total weights pruned in model
        total_level_weights = 0
        prev_layer_neurons = 0
        for layer_name, num_neurons in layer_info.items():
            
            layer_shape = module_dict[layer_name].weight.shape
            out_ch, in_ch, k_h, k_w = layer_shape
            
            neurons_in_curren_layer = len(num_neurons)
            weights_pruned = neurons_in_curren_layer*in_ch*k_h*k_w
            
            total_level_weights += (weights_pruned - prev_layer_neurons*neurons_in_curren_layer)
            prev_layer_neurons = len(num_neurons)
            
        weights_per_level [level] = total_level_weights
    
    return weights_per_level

def num_weights_to_prune(layer_pruned_neurons, model):
    """
    Debug version that prints detailed information about weight calculations

    Calculate the number of weights to be pruned given pruned neuron indices for conv layers.
    Accounts for the fact that outgoing weights of layer n are incoming weights of layer n+1.
    
    Args:
        layer_pruned_neurons (dict): Dictionary mapping layer names to lists of pruned neuron indices
        layer_shapes (dict): Dictionary mapping layer names to weight tensor shapes from named_modules()
                           Shape format: (out_channels, in_channels, kernel_size, kernel_size)
    
    Returns:
        dict: Dictionary containing total weights pruned per layer and summary statistics
    """
    weights_per_level = {}
    module_dict = dict(model.named_modules())
    
    print("\nDetailed Analysis:")
    for level, layer_info in layer_pruned_neurons.items():
        print(f"\nLevel {level}:")
        total_level_weights = 0
        layer_names = list(layer_info.keys())
        
        for i, (layer_name, pruned_neurons) in enumerate(layer_info.items()):
            current_layer = module_dict[layer_name]
            current_shape = current_layer.weight.shape
            out_ch, in_ch, k_h, k_w = current_shape
            
            # Calculate incoming weights
            incoming_weights = len(pruned_neurons) * in_ch * k_h * k_w
            
            print(f"\n  Layer: {layer_name}")
            print(f"  Shape: {current_shape}")
            print(f"  Pruned neurons: {len(pruned_neurons)}")
            print(f"  Incoming weights pruned: {incoming_weights}")
            
            outgoing_weights = 0
            if i < len(layer_names) - 1:
                next_layer_name = layer_names[i + 1]
                next_layer = module_dict[next_layer_name]
                next_shape = next_layer.weight.shape
                outgoing_weights = len(pruned_neurons) * next_shape[0] * k_h * k_w
                print(f"  Next layer: {next_layer_name}")
                print(f"  Next layer shape: {next_shape}")
                print(f"  Outgoing weights pruned: {outgoing_weights}")
            
            layer_total = incoming_weights + outgoing_weights
            print(f"  Total weights pruned for this layer: {layer_total}")
            
            total_level_weights += layer_total
        
        weights_per_level[level] = total_level_weights
        print(f"\nTotal weights pruned at level {level}: {total_level_weights}")
    
    return weights_per_level      



def test(model, test_set, args):
        model.eval()
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        test_acc = 100 * correct / total
        
        return test_acc
        
# print(f"there are no prev pruned ind because level = {level}")

# print(f"prev_percentage = {prev_percentage}", flush = True)
# print(f"current lowest perc = {lowest}, highest = {highest}", flush = True)

# print(f"*******prev ind for {layer}:  \n {prev_ind}", flush= True)
# print(f"*******updated ind for {layer}:  \n {len(updated_unitmems[layer])}", flush =True)
# print(f"total number of neurons {total_neurons-16}")
# print(f"new lowest {new_lowest}, new highest {new_highest}")
# print(f"total acts minus prev acts = {len(acts)}", flush= True)
# print(f"prune_low_count = {prune_low_count}, prune_high_count = {prune_high_count}", flush= True)
# print(f"target neurons before prune_high_count check \n {target_neurons}", flush=True)
# print("prune_high_count !=0", flush=True)
# print(f"target neurons before prune_high_count check: {target_neurons}", flush= True)
