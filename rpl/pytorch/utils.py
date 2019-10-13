import torch
import urllib.request
from tqdm import tqdm

def print_learnable_params(model, freezed_too=False):
    """ Print (learnable) parameters in a given network model

    Args:
    	model: Model you want to print the learned parameters
    	freezed_too: Print the freezed parameters as well
    """
    updated = []
    freezed = []

    for name,param in model.named_parameters():
        if param.requires_grad == True:
            updated.append(name)
        else: 
        	freezed.append(name)

    print("\nFollowing parameters of the model will be updated:")
    for para in updated:
    	print("-", para)

    if freezed_too is True:
	    print("\n Following parameters of the model are freezed:")
	    for para in freezed:
	    	print("-", para)


def predict_probabilities(model, dataloader, beta=1.0, device=torch.device("cpu")):
    """ Predicts data for a given model and a certain beta value (RPL networks)

    Args:
        model: A RPL model 
        dataloader: PyTorch dataloader
        beta: Shifts the prediction of an RPL network
        device: Torch.device to run the calculation
    """
    wrong_probs = []
    correct_probs = []

    for data, target in dataloader:
        # Sends data to GPU
        data = data.to(device)
        target = target.to(device)
        
        # Predict batch
        output = model.forward(data)
        output = torch.exp(beta*output)
        
        # Get max values and indicies 
        preds = torch.max(output, 1)
        
        # Store probabilties in two lists
        for i, pred in enumerate(preds[1]):
            prob_i = preds[0][i].detach().cpu().numpy()

            # get all probs from top1 pred    
            if pred.item() != target[i].item():    
                wrong_probs.append(prob_i)
            else:
                correct_probs.append(prob_i)
    
    return correct_probs, wrong_probs


# https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


# Following function is based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Copyright (c) 2016, Facebook, Inc. All rights reserved.
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Following function is based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Copyright (c) 2016, Facebook, Inc. All rights reserved.
class LogTrainStats():
    """ Container for logging and printing stats in a training process

    Args:
        name: Name used if printing the value
        fmt: Format of the string
    """
    def __init__(self, name, fmt=":.2f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.hist = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.hist.append(val)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)   