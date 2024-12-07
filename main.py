import sys
import os
import re
import copy
import torch
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from ayeshas_code.utils import create_experiment_folder, load_pruned_indices_as_dict, num_weights_to_prune, test
from ayeshas_code.trainer import train
from ayeshas_code.resnets import resnet20
from torchvision.models import resnet50
from ayeshas_code.init import init_dataset
# from ayeshas_code.pruners.prune_unitmem import prune_by_unitmem
from ayeshas_code.pruners.prunitmem2 import prune_by_unitmem
from ayeshas_code.pruners.prune_mag import prune_by_mag
from ayeshas_code.pruners.prune_rand_struct import prune_random_neurons
from ayeshas_code.pruners.prune_l1_struct import prune_by_lowest_l1
from ayeshas_code.pruners.prune_outlier_neuron import prune_outlier_neuron
from torch.utils.data import DataLoader
from ayeshas_code.configs import parse_args


   
def main():
    args = parse_args()
    print(args.exp_desc)
    main_dir = args.dir # results folder
    
    models = {
        "resnet20": resnet20(),
        "resnet50": resnet50()
    }
    model = models.get(args.model)
    model.to("cuda")
    
    # Set pruning method, path and experiment directory
    pruners = {
        "prune_by_unitmem": prune_by_unitmem,
        "prune_by_mag": prune_by_mag,
        "prune_random": prune_random_neurons,
        "prune_lowest_l1": prune_by_lowest_l1,
        "prune_outlier_neuron": prune_outlier_neuron
    }
    pruner = pruners.get(args.pruner)
    prune_path = os.path.join(main_dir, args.pruner_pth)
    print("prune_path = ", prune_path, flush = True)
    result_path = create_experiment_folder(prune_path) # creates "exp_n" in ".. /results/{pruner_pth}"
    
    
    match = re.search(r'(\d+)$', result_path)
    exp_no = match.group(1)
    print(f"exp no. = {exp_no}, model_name = {args.model}, optimizer = {args.optimizer}, \n" 
          f"scheduler = {args.scheduler}, model = {args.init_model}, \n"
          f"batch_size = {args.batch_size}, learning_rate = {args.lr}, \n"
          f"pruning method = {args.prune_by}, results path = {prune_path}", flush= True)
    if args.UM_exp_no != None:
        print(f"UnitMem results for comparison: {args.UM_exp_no}")
    print(result_path)
    
    # Setting X% to pruned each time and densities for each level
    structred_pruning_opts = ['prune_by_unitmem', "prune_random", "prune_lowest_l1", "prune_outlier_neuron" ]
    if args.pruner in structred_pruning_opts:
        prev_pruned_ind = None
        X = args.x
        reduction_factor = 1 - (X / 100)
        density_list = [round(100 * (reduction_factor ** i), 2) for i in range(1,1000) if 100 * (reduction_factor ** i) >= 39]
        sparsity_list = [round(100-x,2) for x in density_list]
        
    elif args.pruner == "prune_by_mag": 
        prev_pruned_ind = load_pruned_indices_as_dict(args.UM_exp_no)
        weights_to_prune_dict = num_weights_to_prune(prev_pruned_ind, model)
        sparsity_list = [weights_to_prune_dict[str(i)] for i in range(len(weights_to_prune_dict.keys()))]
        print(f"Pruning weights in comparison to neurons pruned in exp__{args.UM_exp_no}")
        
    
    print(sparsity_list)
    # Initialize dataset
    train_set, test_set = init_dataset("trainer", args)
    
    # Setting level 
    level = args.lvl
        
    if level == -1:
        # Pretrain if no init model present
        model, ckpt = train(model, level, train_set, test_set, result_path=None, args=args)
        level += 1
    else: 
        init_path = os.path.join(main_dir, args.init_model) 
        print("Path to pretrained model = ",init_path)
        # Start from ckpt, this should be a trained model.
        ckpt = torch.load(init_path)
        print(ckpt.keys())
        model.load_state_dict(ckpt["model_state_dict"])
        pretrained_acc = test(model, test_set, args)
        print(f'Pretrained model accuracy {pretrained_acc}')
        model.train()
    
    
    # Prune and retrain loop
    final_level = len(sparsity_list)-1
    
    # masks = None
    for level, sparsity in enumerate(sparsity_list):
        
        print("_________________________________________", flush = True)
        print(f"current level: {level}/{final_level}", flush = True)
        print("_________________________________________", flush = True)
        # Train and Test sets are NOT wrapped in Dataloader
        # Prune and return dictionary of indices that have been pruned
        updated_prev_pruned_ind = pruner(model, level, sparsity_list, ckpt, result_path, args, prev_pruned_ind)        
        
        # Update prev_pruned_ind for next level if unitmem/random pruning
        if args.pruner != 'prune_by_mag':
            prev_pruned_ind = copy.deepcopy(updated_prev_pruned_ind)
            
        
        # Train pruned model and return retrained one
        model, ckpt = train(model, level, train_set, test_set, result_path, args)
        

        print("_________________________________________", flush = True)
        print(f"Finished pruning and retraining for a total of {level} iteration(s)", flush = True)
        
       

    
if __name__ == "__main__":
    main()   