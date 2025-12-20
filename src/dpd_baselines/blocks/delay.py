import torch
from torch import nn

class Delay(nn.Module):
    def __init__(self, delay):
        super().__init__()
        if not isinstance(delay, int):
            raise TypeError("delay must be int")
        self.delay = delay
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise TypeError("x must consists of Butch and Time dims")
        if not torch.is_complex(x):
            raise TypeError("x must be complex")
        if self.delay == 0:
            return x
        if self.delay < 0:
            zeros = torch.zeros_like(x)
            return torch.cat((x[:, abs(self.delay):], zeros[:, -abs(self.delay):]), dim=1)
        if self.delay > 0:
            zeros = torch.zeros_like(x)
            return torch.cat((zeros[:, :abs(self.delay)], x[:, :-abs(self.delay)]), dim=1)        
