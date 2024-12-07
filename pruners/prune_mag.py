import torch.nn.utils.prune as prune
from ayeshas_code.pruners.prunitmem2 import get_unitmem
 

def prune_by_mag(model, level, sparsity_list, ckpt, result_path, args, prev_pruned_ind):
    """ Pytorch global_structured pruner
    Args:
        model: model to prune,
        level: current iteration of pruning
        sparsity_list: number of weights to prune per level based on neurons pruned in UnitMem pruning
        ckpt: (optional) last model ckpt
        result_path: folder for saving ckpts and unitmems of the model
        args: experiment arguments
        prev_pruned_ind: indices of neurons pruned during unitmem pruning 
        """
    
    model.to("cuda")
    model.eval()
    # Get unitmems for current model
    unitmems = get_unitmem(model, level, result_path, args)
    
    # Get layers to be pruned
    layer_names = [name for name in unitmems.keys()]
    layers_to_prune_list = []
    modules_dict = dict(model.named_modules())
    for name in layer_names:
        layers_to_prune_list.append((modules_dict[name], "weight"))
    
    layers_to_prune = tuple(layers_to_prune_list)
    
    # Get total number of weights to prune
    num_weights_to_prune = sparsity_list[level]
    print(f"total weights pruned: {num_weights_to_prune}")
    
    # Call pruning method 
    prune.global_unstructured(
        layers_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=num_weights_to_prune,
        )
    
    # Print mask hooks 
    for name, mod in model.named_modules():
        if name in unitmems and any(h._tensor_name == "weight" for h in mod._forward_pre_hooks.values()):
            hook = next(h for h in mod._forward_pre_hooks.values() if h._tensor_name == "weight")
            print(f"hook for {name}:{hook}")
    
    return prev_pruned_ind