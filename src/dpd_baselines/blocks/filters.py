import torch
from torch import nn
from typing import Optional, Literal
import torch.nn.functional as F

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

        _, T = x.shape
        h = self.h
        m = h.numel()

        if T == 0:
            return x
        if m == 1:
            return x * h[0]
        pad = m - 1
        x1 = x.unsqueeze(1)
        x_pad = F.pad(x1, (pad, 0))
        y = F.conv1d(x_pad, h.flip(0).view(1, 1, m), bias=None).squeeze(1)
        return y