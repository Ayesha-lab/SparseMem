import numpy as np
import time
import torch
import torch.nn as nn
from torchvision.transforms import v2
from ayeshas_code.utils import append_level_to_json, read_file_to_dict, get_neurons_to_prune
from ayeshas_code.init import init_dataset
from ayeshas_code.CustomPruningMethod import IndexedPruningMethod
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'





def get_unitmem(model, level, result_path, args):
    model.to("cuda")
    model.eval()  
    print(f'Obtaining unit_mem of neurons for level {level}', flush=True)  
    
    # Setting up dataloader with subset of indices
    idx_file = read_file_to_dict(args.indices_file)
    idx_sublist = torch.tensor(idx_file[level])
    
    train_set, _ = init_dataset("unitmem",args)
    subset = torch.utils.data.Subset(train_set, idx_sublist)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)
    print("fetched data and indices", flush=True)
        
    # Create augmentation set
    s=1
    color_jitter = v2.ColorJitter(
            0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s)
    flip = v2.RandomHorizontalFlip()
    augmentations =  v2.Compose(
    [   
        v2.ToPILImage(),
        v2.RandomResizedCrop(size=32),
        v2.RandomApply([flip], p=0.5),
        v2.RandomApply([color_jitter], p=0.9),
        v2.RandomGrayscale(p=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Identifying all ReLU layers
    relu_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.ReLU) and name != "relu"]
    print("Relu_layers size:", len(relu_layers), flush=True)
    
    # Dictionaries for saving data for analysis
    avg_unitmem = {}
    all_layer_activations ={} 

    # Dictionary for computing neurons to prune
    unit_mems = {}
    for name, relu_layer in relu_layers:
        start_time = time.time()
        current_layer_acts = []
        for img_idx, (img, label) in enumerate(iter(dataloader)): 
            img = img.to("cuda")
            single_img_acts = []
            
            for j in range(args.no_augs):
                aug_img = augmentations(img.squeeze(0))
                aug_img = aug_img.unsqueeze(0)
                def hook_fn(module, input, output):
                    
                    act = output.detach().reshape(output.size(1), -1).mean(dim=1) # Reshape and calculate mean. Output shape: torch.size([in_chan, out_chan, k_width, k_height])

                    single_img_acts.append(act) # Act shape: np.array(out_chan,) 
                
                handle = relu_layer.register_forward_hook(hook_fn)
                
                with torch.no_grad():
                    model(img)
                
                handle.remove()
            # single_img_acts shape on axis 0: 10
            current_img_acts = torch.mean(torch.stack(single_img_acts), dim=0)
            current_layer_acts.append(current_img_acts)
            
        # Calculating mean activations and unitmem
        activations = torch.stack(current_layer_acts) # Shape of activations: (1000, 16)
        all_layer_activations[name] = activations 
        maxout = (torch.max(activations, axis=0)[0]).round(decimals=6)  # Shape of maxout:  (16,)
        # print("shape of max_out:", maxout.shape ,"maxout", maxout, flush=True)
        maxposition = torch.argmax(activations,axis=0).cpu() # Shape of maxposition:  (16,)
        # print("shape of max_position:", maxposition.shape , "max_position", maxposition,)
        meanout = (torch.mean(torch.sort(activations, dim=0)[0][:-1], dim=0)).round(decimals=6) # Shape of medianout:  (16,)
    
        denominator = maxout + meanout
        denominator[denominator == 0] += 0.001 # Check for division by 0 and replace by 0.001
        selectivity =  (maxout - meanout) / denominator  # print("----------------- shape of selectivity: ", selectivity.shape) 
        unitmem_neuron_dict = {}
        for i, x in enumerate(selectivity):
            unitmem_neuron_dict[i] = x.item()
           
        conv_name = name.replace("relu", "conv")    
        unit_mems[conv_name] = unitmem_neuron_dict # print("resulting unit_mems", unit_mems, flush=True)
        

        idx_img_highestunitmem = idx_sublist[maxposition] #structure: #idx: neuron idx, unitmem, img idx
        data = []
        for selectivity_val, idx_img_val in zip(selectivity, idx_img_highestunitmem):
            data.append([float(selectivity_val), int(idx_img_val)])
        
        avg_unitmem[name.replace("relu", "conv")] = data
        current_time = time.time()
        process_time = current_time - start_time
        print(f"Processed layer: {name}, time: {process_time:.2f}s", flush = True)
    # append avg_unitmem for this level
    avg_um_path = os.path.join(result_path,"Avg_UnitMems_perlevel.json")
    append_level_to_json(level, avg_unitmem, avg_um_path)
    
    all_activations_path = os.path.join(result_path,"All_activations")
    append_level_to_json(level, all_layer_activations, all_activations_path )
    
    print("level info appended, check file", flush=True)
    
    return unit_mems
  

# This function obtains unit_mems of all neurons for 10000 samples of datapoints
# Then uses these to prune X% of the neurons with the highest unit_mem
# X% is the (1- density), meaning we have X% Sparsity in our model     
def prune_by_unitmem(model, level, sparsity_list, checkpoint, result_path, args, prev_pruned_ind= None):
    # Load last model
    model.to('cuda')
    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    unitmem_pruner = IndexedPruningMethod()
    # Pass the subset for getting unit_mem
    unit_mems = get_unitmem(model, level, result_path, args)
    # Compute how much to sparsify
    sparsity = sparsity_list[level]
    prune_options = {
        "lowest": (args.x, 0, f"pruning lowest {args.x} from unpruned neurons, sparsity: {sparsity}%"),
        "highest": (0, args.x, f"pruning highest {args.x} from unpruned neurons, sparsity: {sparsity}%"),
        "hybrid": (args.x/2, args.x/2, f"pruning {args.x/2}% highest and {args.x/2}% lowest unitmem from unpruned neurons, sparsity: {sparsity}%")
    }
    prune_args = prune_options.get(args.prune_by)  
    indices_to_prune, updated_prev_pruned_ind = get_neurons_to_prune(unit_mems, level, *prune_args[:2], args, prev_pruned_ind)
    print(prune_args[2])
    total_ind_pruned = sum(len(value) for value in indices_to_prune.values())
    print(f'{total_ind_pruned} neurons will be pruned, at indices: \n {indices_to_prune}')  
    
    pruned_ind_path = os.path.join(result_path, "pruned_indices.json")
    # masks_path = os.path.join(result_path, "masks.json")        
    append_level_to_json(level, indices_to_prune, pruned_ind_path)
    # append_level_to_json(level, masks, masks_path)
   
    print(f"hooks after level {level}")
    for layer_name, module in model.named_modules():
            if layer_name in indices_to_prune.keys():
                ind = indices_to_prune[layer_name]
                unitmem_pruner.indices = ind
                unitmem_pruner.apply(module, name="weight", indices = ind)
                
                
                for hook in module._forward_pre_hooks.values():
                    if hook._tensor_name == "weight":  
                        print(f"hook for {layer_name}:{hook}")
                        break
    
    return updated_prev_pruned_ind