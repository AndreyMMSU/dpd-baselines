import torch
import torch.nn as nn
from typing import Optional, Literal

LutInit = Literal["ones", "zeros", "linspace", "identity"]


class Lut1D(nn.Module):
    def __init__(
        self,
        order: int,                      # number of LUT knots
        coeff: Optional[torch.Tensor] = None,
        init: LutInit = "identity",
        trainable: bool = True,
        x_min: float = 0.0,
        x_max: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if not isinstance(order, int):
            raise TypeError("order must be int")
        if order < 2:
            raise ValueError("order must be >= 2 for interpolation")

        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min")

        self.order = order
        self.x_min = float(x_min)
        self.x_max = float(x_max)

        if coeff is not None:
            if not isinstance(coeff, torch.Tensor):
                raise TypeError("coeff must be a torch.Tensor")
            if coeff.ndim != 1:
                raise ValueError("coeff must be 1D tensor")
            if coeff.numel() != order:
                raise ValueError("coeff length must be equal to order")
            c = coeff.to(dtype=dtype)
        else:
            if init == "zeros":
                c = torch.zeros(order, dtype=dtype)
            elif init == "ones":
                c = torch.ones(order, dtype=dtype)
            elif init in ("linspace", "identity"):
                c = torch.linspace(self.x_min, self.x_max, order, dtype=dtype)
            else:
                raise ValueError(f"unknown init: {init}")

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
            raise TypeError("x must be real")
        
        x_clamped = x.clamp(self.x_min, self.x_max)
        scale = (self.order - 1) / (self.x_max - self.x_min)
        pos = (x_clamped - self.x_min) * scale

        i0 = torch.floor(pos).long()
        i1 = torch.clamp(i0 + 1, max=self.order - 1)
        i0 = torch.clamp(i0, max=self.order - 1)

        w = pos - i0.to(pos.dtype)

        y0 = self.coeff[i0] 
        y1 = self.coeff[i1]

        y = (1.0 - w) * y0 + w * y1
        return y