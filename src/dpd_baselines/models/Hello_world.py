import torch
from torch import nn
import torch.nn.functional as F

from dpd_baselines.blocks.delay import Delay
from dpd_baselines.blocks.filters import ComplexFIR
from dpd_baselines.blocks.polynomials import ChebPoly

class Hello_world_model(nn.Module):
    """
    Pipeline: 
    
               |--delay--poly|
    x -> fir --|             + --> y
               |--delay--poly|


    """         

    def __init__(self, 
                 filter_order: int = 5,
                 poly_order0: int = 3, 
                 poly_order1: int = 3):
        super().__init__()
        self.input_layer = ComplexFIR(m=filter_order, init='delta')
        self.d0 = Delay(delay=0)
        self.d1 = Delay(delay=1)
        
        self.poly0 = ChebPoly(order=poly_order0, coeff=torch.zeros(poly_order0, dtype=torch.complex64))
        self.poly1 = ChebPoly(order=poly_order1, coeff=torch.zeros(poly_order0, dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (B, T)")
        if not torch.is_complex(x):
            raise TypeError("x must be complex (B, T)") 
        
        input_layer = self.input_layer(x)

        d0 = self.d0(input_layer)
        d1 = self.d1(input_layer)

        poly0 = self.poly0(d0.abs())
        poly1 = self.poly1(d1.abs())

        y = (poly1*d1 + poly0*d0)

        return y




