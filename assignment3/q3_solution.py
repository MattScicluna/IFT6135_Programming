from PIL import ImageFilter
import random

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Module

from torch import Tensor

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, stop_gradient=True, MLP_mode=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        ablation: if true runs the network without gradient-stopping 
        """
        super(SimSiam, self).__init__()

        self.stop_gradient = stop_gradient
        self.MLP_mode = MLP_mode

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer
        
        if self.MLP_mode=='fixed_random_init':
            # freeze all layers but the last fc
            for param in self.predictor.parameters():
                param.requires_grad = False
            # init the self.predictor layer
            self.predictor[0].weight.data.normal_(mean=0.0, std=0.01)
            self.predictor[3].weight.data.normal_(mean=0.0, std=0.01)
            self.predictor[3].bias.data.zero_()

        elif self.MLP_mode=='no_pred_mlp':
            self.predictor = nn.Identity()
        else:
            pass

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        """ 
        COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.
        Note that the outputs are differnt if stop_gradient is True or False
        """

    def loss (self, p1,p2,z1,z2, similarity_function='CosineSimilarity'):
        """ 
        Input:
            p1,p2,z1,z2: predictors and targets of the network
        Output:
            loss: Simsiam loss 
        """
        """
        COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.
        """

# you might need this function when implementing CosineSimilarity forward function
def bdot(a, b):
    """Performs batch-wise dot product in pytorch"""
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

class CosineSimilarity(Module):
    r"""Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along `dim`.
    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.
    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same number of dimensions as x1, matching x1 size at dimension `dim`,
              and broadcastable with x1 at other dimensions.
        - Output: :math:`(\ast_1, \ast_2)`
    Examples::
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> cos = CosineSimilarity(dim=1, eps=1e-6)
        >>> output = cos(input1, input2)
    """
    __constants__ = ['dim', 'eps']
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """ 
        Input:
            x1,x2: two tensor
        Output:
            cos: cosine similarity between x1,x2
        """
        """
        COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.
        """