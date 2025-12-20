import torch
from torch import nn
from typing import Optional, Literal
class ComplexFIR(nn.Module):
    def __init__(
        self,
        m: int,
        coeff: Optional[torch.Tensor] = None,
        init: Literal["zeros", "delta", "central_delta"] = "delta",
        trainable: bool = True,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()

        if not isinstance(m, int):
            raise TypeError("m must be int")
        if m <= 0:
            raise ValueError("m must be >= 1")

        if coeff is not None:
            if not isinstance(coeff, torch.Tensor):
                raise TypeError("coeff must be a torch.Tensor")
            if coeff.ndim != 1:
                raise ValueError("coeff must be 1D tensor")
            if coeff.numel() != m:
                raise ValueError("coeff length must be m")
            if not torch.is_complex(coeff):
                raise TypeError("coeff must be complex")
            h = coeff.to(dtype=dtype)
        else:
            h = torch.zeros(m, dtype=dtype)
            if init == "zeros":
                pass
            elif init == "delta":
                h[0] = 1
            elif init == "central_delta":
                h[m // 2] = 1
            else:
                raise ValueError("unknown init")

        if trainable:
            self.h = nn.Parameter(h)
        else:
            self.register_buffer("h", h)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (B, T)")
        if not torch.is_complex(x):
            raise TypeError("x must be a complex tensor")

        B, T = x.shape
        h = self.h
        m = h.numel()

        if T == 0:
            return x
        if m == 1:
            return x * h[0]
        pad = m - 1
        x_pad = torch.cat([x.new_zeros((B, pad)), x], dim=1) 
        frames = x_pad.unfold(dimension=1, size=m, step=1)
        h_rev = h.flip(0)  

        y = (frames * h_rev).sum(dim=-1)  
        return y