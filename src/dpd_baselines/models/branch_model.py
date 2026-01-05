import torch
from torch import nn
from typing import Optional, Sequence

from dpd_baselines.blocks.delay import Delay
from dpd_baselines.blocks.filters import ComplexFIR
from dpd_baselines.blocks.polynomials import ChebPoly


class Branch(nn.Module):
    def __init__(
        self,
        delay: int,
        poly_order: int,
        fir_order: Optional[int] = None,
        poly_init: str = "identity",
    ):
        super().__init__()

        if poly_order < 1:
            raise ValueError("poly_order must be >= 1")
        if fir_order is not None and fir_order < 1:
            raise ValueError("fir_order must be >= 1 or None")

        self.delay = Delay(delay=delay)
        self.poly = ChebPoly(order=poly_order, init=poly_init)  # real-valued g(|u|)
        self.fir = ComplexFIR(m=fir_order, init="delta", trainable=True) if fir_order is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.delay(x)
        g = self.poly(u.abs())
        z = u * g
        if self.fir is not None:
            z = self.fir(z)
        return z


def _to_int_list(x: torch.Tensor, name: str) -> list[int]:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D tensor")
    if x.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    return [int(v) for v in x.detach().cpu().tolist()]


def _to_int_list2(x: torch.Tensor, name: str, n0: int, n1: int) -> list[list[int]]:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D tensor")
    if x.shape != (n0, n1):
        raise ValueError(f"{name} must have shape ({n0}, {n1}), got {tuple(x.shape)}")
    return [[int(v) for v in row] for row in x.detach().cpu().tolist()]


class BranchModel(nn.Module):
    """
    2-level tree:
        x
        -> level1: 3 branches
        -> level2: each of 3 outputs goes to 5 leaf branches
        -> sum(15 leaves)
        -> optional out_fir
    """
    def __init__(
        self,
        in_delays: torch.Tensor,        # (3,)
        in_fir_orders: torch.Tensor,    # (3,)
        in_poly_orders: torch.Tensor,   # (3,)
        out_delays: torch.Tensor,       # (3,5)
        out_fir_orders: torch.Tensor,   # (3,5)
        out_poly_orders: torch.Tensor,  # (3,5)
        poly_init: str = "identity",
        out_fir_order: Optional[int] = None,
    ):
        super().__init__()

        n_in = 3
        n_leaf = 5

        in_delays_l = _to_int_list(in_delays, "in_delays")
        in_fir_l = _to_int_list(in_fir_orders, "in_fir_orders")
        in_poly_l = _to_int_list(in_poly_orders, "in_poly_orders")

        if len(in_delays_l) != n_in or len(in_fir_l) != n_in or len(in_poly_l) != n_in:
            raise ValueError("in_* tensors must have length 3")

        out_delays_l = _to_int_list2(out_delays, "out_delays", n_in, n_leaf)
        out_fir_l = _to_int_list2(out_fir_orders, "out_fir_orders", n_in, n_leaf)
        out_poly_l = _to_int_list2(out_poly_orders, "out_poly_orders", n_in, n_leaf)

        self.level1 = nn.ModuleList([
            Branch(delay=in_delays_l[i], poly_order=in_poly_l[i], fir_order=in_fir_l[i], poly_init=poly_init)
            for i in range(n_in)
        ])

        self.level2 = nn.ModuleList([
            nn.ModuleList([
                Branch(
                    delay=out_delays_l[i][j],
                    poly_order=out_poly_l[i][j],
                    fir_order=out_fir_l[i][j],
                    poly_init=poly_init,
                )
                for j in range(n_leaf)
            ])
            for i in range(n_in)
        ])

        self.out_fir = ComplexFIR(m=out_fir_order, init="delta", trainable=True) if out_fir_order else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be torch.Tensor")
        if x.ndim != 2:
            raise ValueError("x must have shape (B, T)")
        if not torch.is_complex(x):
            raise TypeError("x must be complex (B, T)")

        lvl1_out = [br(x) for br in self.level1]

        y = x.new_zeros(x.shape)
        for i, xi in enumerate(lvl1_out):
            for leaf in self.level2[i]:
                y = y + leaf(xi)

        if self.out_fir is not None:
            y = self.out_fir(y)

        return y
