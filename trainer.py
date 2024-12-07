import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import copy
from ayeshas_code.utils import append_level_to_json
from ayeshas_code.init import init_dataset
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def get_optimizer(model, args, checkpoint=None):
    # Create optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Load state if checkpoint exists
    if checkpoint and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return optimizer

def get_scheduler(optimizer, args, start_epoch = -1, checkpoint=None):
    # Create scheduler    
    if args.scheduler == "cosineannealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
    elif args.scheduler == "cosineannealingwarmrestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t_0, T_mult=args.t_mult)
    else:
        scheduler = None

    # Load state if checkpoint exists    
    if checkpoint and scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Step scheduler forward if needed
        if start_epoch >= 0:
            for _ in range(start_epoch):
                scheduler.step()
        
    return scheduler




def train(model, level, train_set, test_set, result_path , args, checkpoint= None):
    device = 'cuda'
    model.to(device)
    train_set, test_set = init_dataset("trainer", args)
    batch_size = args.batch_size
    # Initialize hyperparameters
    if level == -1:
        print("Setting parameters for a pretrained model", flush=True)
        epochs = 500
        patience = float("inf")
        ckpt_path = os.path.join(args.dir, args.init_model) 
        
        
    # If level >= 0, we retrain
    else:  
        print("Pre-trained model already loaded, setting parameters for re-training", flush=True)
        epochs = args.epochs
        patience = 30
        ckpt_path = os.path.join(result_path,f"ckpts/pruned_retrained_model_{level}")


    # Load data with DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,num_workers=4)
    
    # Set up Loss function, Optimizer, Scheduler 
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args, checkpoint)  
    start_epoch = -1  
    scheduler = get_scheduler(optimizer, args, start_epoch, checkpoint)

    # Early stopping setup
    best_loss = float('inf')

    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time = time.time()
        for inputs, labels in train_loader:
            # print("Number of batches insider train_loader loop:", len(train_loader))
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        training_acc = 100 * correct_train / total_train

        # Evaluation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        test_acc = 100 * correct / total
        end_time = time.time()
        process_time = end_time-start_time
        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {training_acc}'
              f', Test Accuracy: {test_acc:.2f}%', f', Validation Loss: {val_loss:.4f}', 
              f', time: {process_time:.2f}s', flush=True)
        
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save the best model
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs', flush=True)
            break
    
    print(f"Training completed. GPU Memory Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB")
    print("Training completed", flush=True)
    
    # Load best model state if found
    if best_model_state is not None:
        model.load_state_dict(best_model_state)   
    # Save final model and hyperparameters
    checkpoint = {
        "model_state_dict": best_model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None
    }
        
    torch.save(checkpoint, ckpt_path)

    return model, checkpoint