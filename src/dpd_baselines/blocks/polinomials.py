import torch
from torch import nn
class ChebPoly(nn.Module):
    def __init__(self, m, coeff):
        super.__init__()
        
        