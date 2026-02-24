import numpy, scipy, torch, dataclasses

from dpd_baselines.signals.filter_design import make_BL


import torch
import scipy.io

from dpd_baselines.signals.filter_design import make_BL


class BlackBoxData80Loader:
    def __init__(self, path, seq_len=2**10, fs=1.2288e6, dtype=torch.complex64):
        try:
            mat = scipy.io.loadmat(path)
        except Exception as e:
            raise ValueError(f"Can't load matfile from {path}: {e}") from e

        for k in ("x", "y", "eRef"):
            if k not in mat:
                raise ValueError(f"In mat file must be '{k}' signal")

        self.seq_len = int(seq_len)
        self.fs = float(fs)

        self.scale: torch.Tensor | None = None
        self.bl_coeff: torch.Tensor | None = None
        self._bl_params: tuple[float, int] | None = None 

        x = torch.as_tensor(mat["x"]).squeeze()
        y = torch.as_tensor(mat["y"]).squeeze()
        eRef = torch.as_tensor(mat["eRef"]).squeeze()

        if x.ndim != 1 or y.ndim != 1 or eRef.ndim != 1:
            raise ValueError("x, y, eRef must be 1D after squeeze()")

        self.x = x.to(dtype=dtype)
        self.y = y.to(dtype=dtype)
        self.eRef = eRef.to(dtype=dtype)

        N = self.x.numel()
        Nw = (N - seq_len)// self.seq_len
        if Nw <= 0:
            raise ValueError(f"Not enough samples: N={N}, seq_len={self.seq_len}")

        n = Nw * self.seq_len
        self.x = self.x[:n].view(Nw, self.seq_len)
        self.y = self.y[:n].view(Nw, self.seq_len)
        self.eRef = self.eRef[:n].view(Nw, self.seq_len)

    def get_signals(self):
        return self.x, self.y, self.eRef

    def apply_bl(self, fc = 0.2e6, numtaps=64):
        if self.bl_coeff is not None:
            old_fc, old_nt = self._bl_params or (None, None)
            raise RuntimeError(f"BL already applied (prev fc={old_fc}, numtaps={old_nt}).")

        y_filt, h = make_BL(self.y, self.fs, fc, numtaps=numtaps)

        if y_filt.dtype != self.y.dtype:
            y_filt = y_filt.to(self.y.dtype)
        if y_filt.device != self.y.device:
            y_filt = y_filt.to(self.y.device)

        self.y = y_filt
        self.bl_coeff = h
        self._bl_params = (float(fc), int(numtaps))
        return h

    def normalize(self):
        if self.scale is not None:
            raise RuntimeError("Signals already normalized (scale is not None).")

        scale = self.x.abs().max()
        self.x = self.x / scale
        self.y = self.y / scale
        self.eRef = self.eRef / scale

        self.scale = scale
        return scale
    

    

         
