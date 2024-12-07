import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration for the model")
    
    # Existing arguments
    parser.add_argument("--model", type=str, default="resnet20", help="Model architecture")
    parser.add_argument("--init_model", type=str, default="init_model_resnet20", help="Initial model name")
    parser.add_argument("--data", type=str, default="cifar_10", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--dir", type=str, default="/home/ayesha/SparseMem/ayeshas_code/results", help="Base directory for results")
    parser.add_argument("--indices_file", type=str, default="ayeshas_code/results/indices_1000_60.csv", help="Indices to use for UnitMem computation")
    parser.add_argument("--UM_batch", type=int, default = 500, help="Batch_size for computing unitmem")
    parser.add_argument("--exp_desc", type= str)
    
    # Optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"], help="Optimizer choice")
    
    # SGD specific arguments
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for SGD optimizer")
    
    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosineannealing", 
                        choices=["cosineannealing", "cosineannealingwarmrestarts"], 
                        help="Scheduler choice")
    
    # CosineAnnealingLR specific arguments
    parser.add_argument("--t_max", type=int, default=50, help="T_max for CosineAnnealingLR scheduler")
    
    # CosineAnnealingWarmRestarts specific arguments
    parser.add_argument("--t_0", type=int, default=5, help="T_0 for CosineAnnealingWarmRestarts scheduler")
    parser.add_argument("--t_mult", type=int, default=1, help="T_mult for CosineAnnealingWarmRestarts scheduler")

    # Pruning arguments
    parser.add_argument("--pruner", type=str, default="prune_by_unitmem", 
                        choices=["prune_by_unitmem", "prune_by_mag", "prune_random", "prune_lowest_l1", "prune_outlier_neuron"], 
                        help="Pruning method: prune_by_unitmem, prune_by_mag")
    parser.add_argument("--prune_by", type=str, default="lowest", 
                        choices=["lowest", "highest", "hybrid"], help = "lowest, highest, hybrid")
    parser.add_argument("--pruner_pth", type=str, default="unitmem", 
                        choices= ["unitmem", "pmag", "random", "lowest_l1", "outliers"], help="Pruner path")
    parser.add_argument("--x", type=float, default=5, help="Percentage to be pruned each time")
    parser.add_argument("--lvl", type=int, default=-1, help="Set to 0 if init model available, -1 otherwise")
    parser.add_argument("--no_augs", type=int, default=10, help="No of Augmentations for computing unitmem")
    
    # Exp number for p_mag to compare with 
    parser.add_argument("--UM_exp_no", type=int, default=None, help ="Experiment number from unitmem results that pmag should compare with")
    
    
    return parser.parse_args()