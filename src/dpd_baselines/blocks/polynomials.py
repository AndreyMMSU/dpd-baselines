import torch
from torch import nn
from typing import Optional, Literal 
class ChebPoly(nn.Module):
    def __init__(
        self,
        order: int,
        coeff: Optional[torch.Tensor] = None,
        trainable: bool = True,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()

        if not isinstance(order, int):
            raise TypeError("order must be int")
        if order <= 0:
            raise ValueError("order must be >= 1")
        self.order = order

        if coeff is not None:
            if not isinstance(coeff, torch.Tensor):
                raise TypeError("coeff must be a torch.Tensor")
            if coeff.ndim != 1:
                raise ValueError("coeff must be 1D tensor")
            if coeff.numel() != order:
                raise ValueError("coeff length must be equal to order")
            if not torch.is_complex(coeff):
                raise TypeError("coeff must be complex")
            c = coeff.to(dtype=dtype)
        else:
            c = torch.zeros(order, dtype=dtype)
            c[0] = 1 

        if trainable:
            self.coeff = nn.Parameter(c)
        else:
            self.register_buffer("coeff", c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            raise TypeError("x must be a torch.Tensor, got None")
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 2:
            raise ValueError("x must have shape (B, T)")
        if x.is_complex():
            raise TypeError("x must be real for Chebyshev polynomial evaluation")
        if x.abs().max() > 1:
            x = x.clamp(-1.0, 1.0)
        
        c = self.coeff
        K = self.order
        if K == 0:
            raise RuntimeError("coeff is empty (order must be >= 1)")
        T_prev = torch.ones_like(x)          
        g = c[0] * T_prev                    
        if K == 1:
            return g

        T_curr = x                        
        g = g + c[1] * T_curr               

        for k in range(2, K):
            T_next = (2.0 * x) * T_curr - T_prev
            g = g + c[k] * T_next
            T_prev, T_curr = T_curr, T_next

        return g
                
