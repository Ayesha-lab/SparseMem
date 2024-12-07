import torch
import torch.nn as nn
import numpy as np
import time
import os
from collections import defaultdict
from pruners.prune_unitmem import get_unitmem

def get_conv_prune_connections(model, prune_percentage):
    # Get all convolutional layers, excluding those with "downsample" in the name
    conv_layers = [(name, module) for name, module in model.named_modules() 
                   if isinstance(module, nn.Conv2d) and "downsample" not in name]
    
    # Gather all weights
    all_weights = []
    for _, layer in conv_layers:
        all_weights.append(layer.weight.data.abs().flatten())
    
    all_weights = torch.cat(all_weights)
    
    # Calculate number of weights to prune
    num_weights_to_prune = int(len(all_weights) * prune_percentage / 100)
    
    # Get the threshold value
    threshold = all_weights.kthvalue(num_weights_to_prune).values.item()
    
    # Gather connections to prune
    connections_to_prune = []
    for layer_idx, (layer_name, layer) in enumerate(conv_layers):
        weight_tensor = layer.weight.data.abs()
        below_threshold = weight_tensor <= threshold
        
        # Get indices of weights below threshold
        prune_indices = below_threshold.nonzero(as_tuple=False)
        
        # Store connections as (layer_name, layer_idx, output_channel, input_channel, kernel_h, kernel_w)
        for idx in prune_indices:
            connections_to_prune.append((layer_name, layer_idx, idx[0].item(), idx[1].item(), idx[2].item(), idx[3].item()))
    
    return connections_to_prune, conv_layers

def prune_by_mag(model, train_set, level, sparsity, checkpoint, result_path, args, prev_pruned_ind=None):
    
    model.to('cuda')
    # Load last model
    model.load_state_dict(checkpoint['model_state_dict'])
    # Pass the subset for getting unit_mem
    if level >= 0:
        unit_mems = get_unitmem(model, level, train_set, result_path)
    
    if sparsity <= 0:
        print(f"nothing to prune, sparsity; {sparsity}", flush=True)
    else:
        print(f"identifying the lowest {sparsity}% weights to prune", flush=True)
        
        connections, conv_layers = get_conv_prune_connections(model, sparsity)
        
        # Group connections by layer
        layer_connections = defaultdict(list)
        for conn in connections:
            layer_connections[conn[0]].append(conn[1:])
        
        # Prune weights layer by layer
        for layer_name, conns in layer_connections.items():
            start_time = time.time()
            print(f"UnitMem scores: {unit_mems[layer_name]}")
            # Find the corresponding layer in conv_layers
            layer_idx = next(idx for idx, (name, _) in enumerate(conv_layers) if name == layer_name)
            layer = conv_layers[layer_idx][1]
            
            # Prune all connections for this layer
            for _, out_channel, in_channel, h, w in conns:
                layer.weight.data[out_channel, in_channel, h, w] = 0
            
            end_time = time.time()
            prune_time = end_time - start_time
            print(f"pruning weights in {layer_name} complete, time: {prune_time:.2f}s", flush=True)
    
    # Create a new checkpoint with the pruned model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
    }
    path = os.path.join(result_path,f"ckpts/pruned_model_{level}")
    torch.save(checkpoint, path)
    
    return model, checkpoint, path