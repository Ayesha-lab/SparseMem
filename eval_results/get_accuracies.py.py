import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import torchvision
import os
import glob
import re
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from ayeshas_code.resnets import resnet20

def apply_pruning_masks(model, checkpoint_path):
    """Apply pruning masks from checkpoint to model."""
    checkpoint = torch.load(checkpoint_path)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    conv_names = [name for name, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
    modules = dict(model.named_modules())
    
    # Apply identity pruning and load masks in one pass
    for name in conv_names:
        module = modules[name]
        prune.identity(module, 'weight')
        
        # Load weights and masks
        module.weight_orig.data.copy_(checkpoint[f"{name}.weight_orig"])
        module.weight_mask.data.copy_(checkpoint[f"{name}.weight_mask"])
    
    # Load remaining parameters
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() 
                      if k in model_dict and not (k.endswith('_orig') or k.endswith('_mask'))}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model

@torch.no_grad()
def compute_accuracy(model, loader, device):
    """Compute accuracy efficiently."""
    model.eval()
    correct = total = 0
    
    for inputs, labels in loader:
        outputs = model(inputs.to(device))
        correct += outputs.argmax(1).eq(labels.to(device)).sum().item()
        total += labels.size(0)
    
    return (correct / total) * 100

@lru_cache(maxsize=None)
def get_dataloaders(data_path, batch_size):
    """Cache dataloaders to avoid recreating them."""
    transforms = v2.Compose([
        v2.Resize(32),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5] * 3, [0.5] * 3)
    ])
    
    datasets = {
        'train': torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transforms),
        'test': torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transforms)
    }
    
    return {split: DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for split, dataset in datasets.items()}

def evaluate_checkpoints(config):
    """Evaluate all checkpoints with given configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_folder = os.path.join(config['exp_folder'], 'ckpts')
    checkpoints = glob.glob(os.path.join(ckpt_folder, 'pruned_retrained_model_*'))
    model = config['model_class']().to(device)
    loaders = get_dataloaders(config['data_path'], config['batch_size'])
    
    results = []
    for ckpt_path in tqdm(sorted(checkpoints, key=lambda x: int(re.search(r'(\d+)', x).group()))):
        iteration = int(re.search(r'model_(\d+)', ckpt_path).group(1))
        model = apply_pruning_masks(model, ckpt_path)
        
        # Calculate metrics
        metrics = {
            'iteration': iteration,
            'sparsity': round(100 - (100 * (config['reduction_factor'] ** (iteration + 1))), 2),
            'train_acc': round(compute_accuracy(model, loaders['train'], device), 2),
            'test_acc': round(compute_accuracy(model, loaders['test'], device), 2)
        }
        metrics['generalization_gap'] = round(metrics['train_acc'] - metrics['test_acc'], 2)
        results.append(metrics)
    
    return pd.DataFrame(results).sort_values('iteration')

# Usage
if __name__ == "__main__":
    config = {
        'exp_num': 10,
        'exp_folder': "/home/ayesha/SparseMem/eval_neurons/results_all_20112024/unitmem/exp__10",
        'data_path': "/home/ayesha/SparseMem/ayeshas_code/data/cifar-10-python",
        'batch_size': 256,
        'reduction_factor': 0.95,  # 1 - (5/100)
        'model_class': resnet20
    }
    
    results_df = evaluate_checkpoints(config)
    output_path = os.path.join(config['exp_folder'], 'pruning_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print("\nResults summary:")
    print(results_df)