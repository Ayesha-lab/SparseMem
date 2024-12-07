import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, Dataset 
from torchvision.transforms import v2
import torch.nn.functional as F
from ayeshas_code.utils import append_level_to_json, read_file_to_dict, get_neurons_to_prune
from ayeshas_code.init import init_dataset
from ayeshas_code.CustomPruningMethod import IndexedPruningMethod
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def get_outlierneuron(model, level, result_path, args):
    model.to("cuda")
    model.eval()  
    UM_batch = args.UM_batch
    print(f'Obtaining unit_mem of neurons for level {level}', flush=True)  

    class AugmentedDataset(Dataset):
        def __init__(self, original_dataset, transforms, num_augs=10):
            self.dataset = original_dataset
            self.transforms = transforms
            self.num_augs = num_augs
            
        def __getitem__(self, idx):
            real_idx = idx // self.num_augs
            img, label = self.dataset[real_idx]
            img = self.transforms(img)
            return img, label
        
        def __len__(self):
            return len(self.dataset) * self.num_augs

    # Create augmentation set
    s = 1
    augmentations = v2.Compose([
        v2.ToPILImage(),
        v2.RandomResizedCrop(size=32),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s),
        v2.RandomGrayscale(p=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Setting up dataloader with subset of indices
    idx_file = read_file_to_dict(args.indices_file)
    idx_sublist = torch.tensor(idx_file[level])
    
    # Setup dataset and dataloader
    train_set, _ = init_dataset("unitmem", args)
    subset = torch.utils.data.Subset(train_set, idx_sublist)
    augmented_dataset = AugmentedDataset(subset, augmentations, num_augs=args.no_augs)
    dataloader = DataLoader(
        augmented_dataset, 
        batch_size=UM_batch * args.no_augs,  # Increased batch size to account for augmentations
        shuffle=False,
        num_workers=4
    )
    print(f"len of dataloader {len(dataloader)}", flush=True)
    print("fetched data and indices", flush=True)

    # Identifying all Batch_norm layers
    bn_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d) and name != "bn1"]
    print("bn_layers size:", len(bn_layers), flush=True)

    # Dictionaries for saving data for analysis
    avg_unitmem = {}
    all_layer_activations = {}
    unit_mems = {}

    for name, bn_layer in bn_layers:
        start_time = time.time()
        current_layer_acts = []
        
        for img_batch, label in iter(dataloader):
            img_batch = img_batch.to("cuda")
            batch_size = img_batch.size(0)
            
            # Reshape batch to group augmentations
            img_batch = img_batch.view(-1, args.no_augs, *img_batch.shape[1:])
            
            current_aug_acts = []
            def hook_fn(module, input, output):
                # Shape: (batch_size, channels, feature_maps) -> (batch_size, channels)
                relu_output = F.relu(output)
                act = relu_output.detach().reshape(output.size(0), output.size(1), -1).mean(dim=2)
                current_aug_acts.append(act)
            
            handle = bn_layer.register_forward_hook(hook_fn)
            
            with torch.no_grad():
                # Process the entire batch at once
                output = model(img_batch.view(-1, *img_batch.shape[2:]))
            
            handle.remove()
            
            # Reshape activations to group by original image
            acts = torch.stack(current_aug_acts).squeeze(0)
            acts = acts.view(-1, args.no_augs, acts.shape[1])
            
            # Average across augmentations
            mean_batch_acts = torch.mean(acts, dim=1)
            current_layer_acts.append(mean_batch_acts)

        # Calculating mean activations and unitmem
        activations = torch.cat(current_layer_acts, dim=0)
        all_layer_activations[name] = activations
        # Get Max and respective image
        maxout = (torch.max(activations, axis=0)[0]).round(decimals=6)
        maxposition = torch.argmax(activations, axis=0).cpu()
        # Get second max and respective image
        values, indices = torch.topk(activations, k=2, dim=0)
        second_max = values[1]
        second_max_pos = indices[1]
        

        denominator = maxout + second_max
        denominator[denominator == 0] += 0.001
        selectivity = (maxout - second_max) / denominator

        outlier_neuron_dict = {}
        for i, x in enumerate(selectivity):
            outlier_neuron_dict[i] = x.item()
           
        conv_name = name.replace("bn", "conv")    
        unit_mems[conv_name] = outlier_neuron_dict

        data = list(zip(map(float, selectivity), map(int, maxposition), map(int, second_max_pos)))
        avg_unitmem[name.replace("bn", "conv")] = data
        
        current_time = time.time()
        process_time = current_time - start_time
        print(f"Processed layer: {name}, time: {process_time:.2f}s", flush=True)

    # Save results
    avg_um_path = os.path.join(result_path, "Avg_UnitMems_perlevel.json")
    append_level_to_json(level, avg_unitmem, avg_um_path)
    
    all_activations_path = os.path.join(result_path, "All_activations")
    append_level_to_json(level, all_layer_activations, all_activations_path)
    
    print("level info appended, check file", flush=True)
    
    return unit_mems
  

# This function obtains unit_mems of all neurons for 10000 samples of datapoints
# Then uses these to prune X% of the neurons with the highest unit_mem
# X% is the (1- density), meaning we have X% Sparsity in our model     
def prune_outlier_neuron(model, level, sparsity_list, checkpoint, result_path, args, prev_pruned_ind= None):
    # Load last model
    model.to('cuda')
    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    unitmem_pruner = IndexedPruningMethod()
    # Pass the subset for getting unit_mem
    unit_mems = get_outlierneuron(model, level, result_path, args)
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