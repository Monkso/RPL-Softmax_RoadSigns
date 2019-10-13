# Implementation is based on the FGSM PyTorch tutorial.
# For more information see: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

import warnings
import numpy as np
import torch

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def run_attack(model, test_loader, criterion, epsilon, beta=1.0, device=torch.device("cpu")):
    # Accuracy counter
    correct = 0
    adv_examples = []
    output_props = []
    output_prop_wrong = []

    # Loop over all examples in test set
    for data, target in test_loader:
        # Move to device
        data = data.to(device)
        target = target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # If the prediction was correct run a FGSM attack
        # Backprop
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # Call FGSM Attack - alter data and run the forward path again
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        
        # Check the prediction again
        final_pred = output.max(1, keepdim=True)[1] 
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Wrong prediction after attack
            if epsilon>0.01:
                # Get max probability before perturbation
                output_prop = model.probabilities(data, beta).detach().cpu().numpy().max(axis=1)
                output_props.append(output_prop)
                
                # Get max probability after perturbation
                output_prop_wrong_ = model.probabilities(perturbed_data, beta).detach().cpu().numpy().max(axis=1)
                output_prop_wrong.append(output_prop_wrong_)
                
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    
    # Convert propps to numpy arrays
    output_prop_wrong = np.array(output_prop_wrong)
    output_props = np.array(output_props)

    # Catch the warning for eps=0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output_props = output_props.mean()
        output_prop_wrong = output_prop_wrong.mean()

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, output_props, output_prop_wrong
