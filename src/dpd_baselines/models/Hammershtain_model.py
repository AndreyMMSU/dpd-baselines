import torch
from torch import nn
from typing import List

from dpd_baselines.blocks.delay import Delay
from dpd_baselines.blocks.filters import ComplexFIR
from dpd_baselines.blocks.polynomials import ChebPoly

class Branch(nn.Module):
    def __init__(self, delay, poly_order):
        super().__init__()

        self.d1 = Delay(delay=delay)
        self.d2 = Delay(delay=delay)
        self.p = ChebPoly(order=poly_order)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError('x must be a torch.Tensor')
        if x.ndim != 2:
            raise ValueError('x must have 2 dims (B, T)')
        
        return self.p(self.d2(x).abs())*self.d1(x)

class Hammershtain(nn.Module):
    """
    Pipeline: 
    
         |--delay--poly|
    x -> |  .........  + --> fir --> BL --> y
         |--delay--poly|

    """         

    def __init__(self, 
                 poly_bank_delays: List[int],
                 poly_bank_orders: List[int],
                 filter_order_out: int = 5,
                 BL_coeff: torch.Tensor = torch.tensor([1 + 0j])):
        super().__init__()
        
        if not isinstance(poly_bank_orders, list):
            raise TypeError('poly_bank_orders must be list')
        if not isinstance(poly_bank_delays, list):
            raise TypeError('poly_bank_orders must be list')
        if len(poly_bank_orders) != len(poly_bank_delays):
            raise ValueError("len(poly_bank_orders) != len(poly_bank_delays)")
        self.N = len(poly_bank_orders)

        self.fir = ComplexFIR(m=filter_order_out, init='delta')
        self.poly_bank = nn.ModuleList([Branch(poly_bank_delays[i], poly_bank_orders[i])
            for i in range(self.N)])
        self.BL = ComplexFIR(m=BL_coeff.shape[0], coeff=BL_coeff, trainable=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (B, T)")
        if not torch.is_complex(x):
            raise TypeError("x must be complex (B, T)") 
        
        s = self.poly_bank[0](x) 
        for i in range(1, self.N):
            s = s + self.poly_bank[i](x) 
        y = self.BL(self.fir(s))

        return y




