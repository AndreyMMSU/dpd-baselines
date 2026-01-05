import torch
from torch import nn
import torch.nn.functional as F

from dpd_baselines.blocks.delay import Delay
from dpd_baselines.blocks.filters import ComplexFIR
from dpd_baselines.blocks.polynomials import ChebPoly

class branch_model(nn.Module):
    """
    Pipeline: 
    
               |--delay--poly|
    x -> fir --|             + --> fir --> y
               |--delay--poly|
               |--delay--poly|
               |--delay--poly|

    """         

    def __init__(self, 
                 filter_order: int = 5,
                 poly_order0: int = 3, 
                 poly_order1: int = 3,
                 poly_order2: int = 3,
                 poly_order3: int = 3):
        super().__init__()
        self.input_layer = ComplexFIR(m=filter_order, init='delta')
        self.d0 = Delay(delay=0)
        self.d1 = Delay(delay=1)
        self.d2 = Delay(delay=-1)
        self.d3 = Delay(delay=2)    

        self.poly0 = ChebPoly(order=poly_order0, coeff=torch.zeros(poly_order0, dtype=torch.complex64))
        self.poly1 = ChebPoly(order=poly_order1, coeff=torch.zeros(poly_order1, dtype=torch.complex64))
        self.poly2 = ChebPoly(order=poly_order2, coeff=torch.zeros(poly_order2, dtype=torch.complex64))
        self.poly3 = ChebPoly(order=poly_order3, coeff=torch.zeros(poly_order3, dtype=torch.complex64))

        self.endfir = ComplexFIR(m = 3, init='delta')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (B, T)")
        if not torch.is_complex(x):
            raise TypeError("x must be complex (B, T)") 
        
        input_layer = self.input_layer(x)

        d0 = self.d0(input_layer)
        d1 = self.d1(input_layer)
        d2 = self.d1(input_layer)
        d3 = self.d1(input_layer)

        poly0 = self.poly0(d0.abs())
        poly1 = self.poly1(d1.abs())
        poly2 = self.poly2(d2.abs())
        poly3 = self.poly3(d3.abs())

        y = (poly1*d1 + poly0*d0 + poly2*d2 + poly3*d3)
        end = self.endfir(y)
        
        return end




