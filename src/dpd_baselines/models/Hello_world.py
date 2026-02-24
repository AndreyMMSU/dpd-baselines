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
    x -> fir --|             + --> fir --> BL --> y
               |--delay--poly|


    """         

    def __init__(self, 
                 filter_order_in: int = 5,
                 filter_order_out: int = 5,
                 poly_order0: int = 3, 
                 poly_order1: int = 3,
                 BL_coeff: torch.Tensor = torch.tensor([1 + 0j])):
        super().__init__()
        self.input_layer = ComplexFIR(m=filter_order_in, init='delta')
        self.d0 = Delay(delay=0)
        self.d1 = Delay(delay=1)

        coeff_init_poly0 = torch.zeros(poly_order0, dtype=torch.complex64)
        coeff_init_poly0[1] = 0+0j
        coeff_init_poly1 = torch.zeros(poly_order1, dtype=torch.complex64)
        coeff_init_poly1[1] = 0+0j

        self.poly0 = ChebPoly(order=poly_order0, coeff=coeff_init_poly0)
        self.poly1 = ChebPoly(order=poly_order1, coeff=coeff_init_poly1)

        self.output_fir = ComplexFIR(m=filter_order_out, init='delta')
        self.BL = ComplexFIR(m=BL_coeff.shape[0], coeff=BL_coeff, trainable=False)


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

        y = self.BL(self.output_fir((poly1*d1 + poly0*d0)))


        return y




