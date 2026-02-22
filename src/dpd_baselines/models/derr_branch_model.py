import torch
from torch import nn
from typing import Optional, Sequence, Dict, Tuple

from dpd_baselines.blocks.delay import Delay
from dpd_baselines.blocks.filters import ComplexFIR


def _check_bt_complex(x: torch.Tensor, name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape (B, T)")
    if not torch.is_complex(x):
        raise TypeError(f"{name} must be complex (B, T)")


def dv_power(u: torch.Tensor, h: torch.Tensor, p: int, eps: float = 1e-12) -> torch.Tensor:
    """
    v(u)=u|u|^{p-1}
    Dv(u)[h] = (p+1)/2 * |u|^{p-1} h + (p-1)/2 * u^2 |u|^{p-3} h*
    Real-Frechet derivative in complex form (h and h*).
    """
    if p < 1 or (p % 2) == 0:
        raise ValueError("p must be odd and >= 1")

    au = u.abs().clamp_min(eps)
    term1 = ((p + 1) / 2.0) * (au ** (p - 1)) * h
    if p == 1:
        term2 = torch.zeros_like(term1)
    else:
        term2 = ((p - 1) / 2.0) * (u ** 2) * (au ** (p - 3)) * torch.conj(h)
    return term1 + term2


class PowerFIRBranch(nn.Module):
    """
    One minimal GMP-like branch with:
      u = Delay(x)
      v = u |u|^{p-1}
      s = FIR(v) (optional)
      out = a * s  (complex adaptive gain)

    Parameters are shared when we reuse the same branch FIR+gain for derived vectors.
    """
    def __init__(
        self,
        delay: int,
        power_order: int,
        fir_order: Optional[int] = None,
    ):
        super().__init__()

        if delay < 0:
            raise ValueError("delay must be >= 0")
        if power_order < 1 or (power_order % 2) == 0:
            raise ValueError("power_order must be odd and >= 1")
        if fir_order is not None and fir_order < 1:
            raise ValueError("fir_order must be >= 1 or None")

        self.power_order = int(power_order)
        self.delay = Delay(delay=delay)
        self.fir = ComplexFIR(m=fir_order, init="delta", trainable=True) if fir_order is not None else None

        self.a_re = nn.Parameter(torch.tensor(1.0))
        self.a_im = nn.Parameter(torch.tensor(0.0))

    def a(self) -> torch.Tensor:
        return torch.complex(self.a_re, self.a_im)

    def base(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (u, v) where:
          u = Delay(x)
          v = u |u|^{p-1}
        """
        _check_bt_complex(x, "x")
        u = self.delay(x)
        v = u * (u.abs() ** (self.power_order - 1))
        return u, v

    def linear(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply shared linear part after nonlinearity: FIR (if any) and gain.
        """
        _check_bt_complex(z, "z")
        y = self.fir(z) if self.fir is not None else z
        return self.a() * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.base(x)
        return self.linear(v)


class FilteredGMP_DN(nn.Module):
    """
    z1 = N(x) = sum_i Branch_i(x)
    z2 = DN[x](z1) computed analytically with shared branch parameters:
         DN[x](z1) = sum_i a_i * FIR_i( Dv_i(u_i)[h_i] ),
         where u_i=Delay_i(x), h_i=Delay_i(z1)
    yhat = z1 + g1*z2

    Returns yhat; use forward_with_debug for (z1,z2).
    """
    def __init__(
        self,
        orders: Sequence[int],
        delays: Sequence[int],
        fir_orders: Sequence[Optional[int]],
        eps: float = 1e-12,
    ):
        super().__init__()

        if not (len(orders) == len(delays) == len(fir_orders)):
            raise ValueError("orders/delays/fir_orders must have same length")
        if len(orders) == 0:
            raise ValueError("must have at least 1 branch")

        self.eps = float(eps)
        self.orders = [int(p) for p in orders]

        self.branches = nn.ModuleList([
            PowerFIRBranch(delay=int(delays[i]), power_order=int(orders[i]), fir_order=fir_orders[i])
            for i in range(len(orders))
        ])

        self.g1_re = nn.Parameter(torch.tensor(0.0))
        self.g1_im = nn.Parameter(torch.tensor(0.0))

    def g1(self) -> torch.Tensor:
        return torch.complex(self.g1_re, self.g1_im)

    def warmup(self) -> int:
        # Safe-ish warmup for metrics: max delay + max fir order
        max_delay = 0
        max_fir = 0
        for br in self.branches:
            # Delay object stores delay; assume attribute exists
            if hasattr(br.delay, "delay"):
                max_delay = max(max_delay, int(br.delay.delay))
            # ComplexFIR stores m; assume attribute exists
            if br.fir is not None and hasattr(br.fir, "m"):
                max_fir = max(max_fir, int(br.fir.m))
        return max_delay + max_fir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yhat, _ = self.forward_with_debug(x)
        return yhat

    def forward_with_debug(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _check_bt_complex(x, "x")

        # z1 = sum branches
        z1 = x.new_zeros(x.shape)
        for br in self.branches:
            z1 = z1 + br(x)

        # z2 = sum branches of analytic Dv(u)[h] passed through same FIR+gain
        z2 = x.new_zeros(x.shape)
        for br, p in zip(self.branches, self.orders):
            u = br.delay(x)
            h = br.delay(z1)
            dv = dv_power(u=u, h=h, p=p, eps=self.eps)
            z2 = z2 + br.linear(dv)

        yhat = z1 + self.g1() * z2
        dbg = {"z1": z1, "z2": z2, "yhat": yhat}
        return yhat, dbg
