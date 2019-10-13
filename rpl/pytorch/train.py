import copy
import time

import torch

from rpl.pytorch.utils import LogTrainStats
from rpl.pytorch.utils import accuracy

def train_imagenet(model, dataloaders, criterion, optimizer, num_epochs=10, device=torch.device("cpu"), verbose=False):
    """ Main training loop for ImageNet dataset (VGG network)
    
    Args:
        model: Radial Decision Network
        dataloaders: Data using torch.utils.data.DataLoader
        criterion: Loss metric
        optimizer: Optimizer using torch.optim
        num_epochs: Number of training epochs
        verbose: If True, Batch stats will be printed
    Returns:
        model:  Model with the best accuracy
        (train_stats): loss, top1/5 acc for each batch and validation
        (val_stats): top1/5 acc for each batch and validation
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc1 = 0.0

    # Container to allocate train statistics
    train_losses = LogTrainStats("loss", ":.4e")
    train_top1 = LogTrainStats("top-1")
    train_top5 = LogTrainStats("top-5")
    train_epoch_top1 = LogTrainStats("top-1")
    train_epoch_top5 = LogTrainStats("top-5")
    val_top1 = LogTrainStats("top-1")
    val_top5 = LogTrainStats("top-5")
    
    # Start timer for the whole training
    train_start_time = time.time()
    print("\nLet's start training... \n")

    for epoch in range(num_epochs):
        # Start timer for the epoch
        epoch_start_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  
            else:
                model.eval()   

            # Iterate over data.
            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                # Move to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Log Batch Statistics
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                if phase == "train":
                    train_top1.update(acc1[0], inputs.size(0))
                    train_top5.update(acc5[0], inputs.size(0))
                    train_losses.update(loss.item())
                else:
                    val_top1.update(acc1[0], inputs.size(0))
                    val_top5.update(acc5[0], inputs.size(0))

                # Batch logging
                if verbose is True and phase == "train":
                    if step % 100 == 0:
                        print("E:[{}] - B:[{}/{}] || {} || {} || {} ".format(epoch, 
                                                                             step, 
                                                                             len(dataloaders[phase]), 
                                                                             train_losses, 
                                                                             train_top1, 
                                                                             train_top5
                                                                            ))
                        
            # Log epoch statistics
            if phase == "train":
                train_epoch_top1.update(acc1[0])
                train_epoch_top5.update(acc5[0])
                                        
            # Calcualte epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch logging
            if phase == "train":
                print("T-Ep[{}] || {:.0f}m {:.0f}s || {} || {} || {} ".format(epoch, 
                                                                              epoch_time//60,
                                                                              epoch_time%60, 
                                                                              train_losses, 
                                                                              train_top1, 
                                                                              train_top5
                                                                             ))
            else:
                print("V-Ep[{}] || {:.0f}m {:.0f}s || {} || {} ".format(epoch, 
                                                                        epoch_time//60,
                                                                        epoch_time%60, 
                                                                        val_top1, 
                                                                        val_top5
                                                                       ))
            
            # Deep copy opf the best top-1 model
            if phase == "val" and val_top1.val >= best_acc1:
                best_acc1 = val_top1.val
                best_model_wts = copy.deepcopy(model.state_dict())
    
    # Print overall training time and best top-1 acc    
    train_time = time.time() - train_start_time
    print("\n...training complete in {:.0f}m {:.0f}s".format(train_time // 60, train_time % 60))
    print("\nBest Validtion Acc (top-1): {:.2f}".format(best_acc1))
    
    # Load best model weights and return it with some training statistics
    model.load_state_dict(best_model_wts)
    return model, best_acc1, (train_losses.hist, train_top1.hist, train_top5.hist, train_epoch_top1.hist, train_epoch_top5.hist), (val_top1.hist, val_top5.hist)