import torch
from typing import Tuple

class Prod(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *signals: torch.Tensor):
        if len(signals) == 0:
            raise ValueError("Prod requires at least one tensor")
        
        for sig in signals:
            if not isinstance(sig, torch.Tensor):
                raise TypeError("All inputs must be torch.Tensor")
            
        for sig in signals:
            if sig.ndim != 2:
                raise ValueError('All x must be 2D torch.Tensor (B, T)')
        prod = signals[0]
        for sig in signals[1:]:
            prod = prod * sig
        return prod

class Sum(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *signals):
        if len(signals) == 0:
            raise ValueError("Sum requires at least one tensor")
        for sig in signals:
            if not isinstance(sig, torch.Tensor):
                raise TypeError("All inputs must be torch.Tensor")
        for sig in signals:
            if sig.ndim != 2:
                raise ValueError('All x must be 2D torch.Tensor (B, T)')
            
        _sum = signals[0]
        for sig in signals[1:]:
            _sum = _sum + sig
        return _sum
    
class Conj(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be torch.Tensor")
        if x.ndim != 2:
            raise ValueError('x must be 2D torch.Tensor (B, T)')
        return x.conj()

class Abs(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be torch.Tensor")
        if x.ndim != 2:
            raise ValueError('x must be 2D torch.Tensor (B, T)')
        return x.abs()
    
