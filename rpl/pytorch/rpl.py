import torch
import torch.nn as nn

class RadialPredictionLayer(torch.nn.Module):
    """ The RPL classification layer with fixed prototypes
    """
    def __init__(self, in_features, out_features):
        super(RadialPredictionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features        
        # Initialize prototype unit vectors on each axis 
        self.prototypes = nn.Parameter(torch.diag((torch.ones(self.in_features))), requires_grad=False)


    def forward(self, x):
        # Calculate the Euclidian distance between row vectors and prototypes
        return ( - (((x[:, None,:] - self.prototypes[None,:,:])**2).sum(dim=2)).sqrt())


    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)




class RadialPredictionLoss(torch.nn.Module):
    """ Loss for networks using Radial Predition Layzer
    """
    def __init__(self):
        super(RadialPredictionLoss, self).__init__()
    
    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]
        loss = - predictions[range(batch_size), targets]
        return loss.sum()/batch_size 