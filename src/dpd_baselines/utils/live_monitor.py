from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_psd_welch(
    x: np.ndarray,
    nfft: int = 4096,
    hop: Optional[int] = None,
    window: str = "hann",
    fs: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x = _to_numpy(x)
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if hop is None:
        hop = nfft // 2

    if window != "hann":
        raise ValueError("only hann window supported for now")

    win = np.hanning(nfft).astype(np.float64)
    win_pow = np.sum(win * win) + 1e-30

    psd_acc = np.zeros(nfft, dtype=np.float64)
    cnt = 0

    N = x.shape[0]
    for s in range(0, N - nfft + 1, hop):
        seg = x[s : s + nfft] * win
        X = np.fft.fft(seg, n=nfft)
        X = np.fft.fftshift(X)
        P = (np.abs(X) ** 2) / win_pow
        psd_acc += P
        cnt += 1

    psd = psd_acc / max(1, cnt)

    d = 1.0 if fs is None else (1.0 / float(fs))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=d))
    return f, psd


def psd_to_db(psd: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    psd = np.asarray(psd)
    return 10.0 * np.log10(psd + eps)


def normalize_psd_to_0db(psd_db: np.ndarray, ref_psd_db: np.ndarray) -> np.ndarray:
    shift = float(np.max(ref_psd_db))
    return psd_db - shift


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class LiveMonitor:
    nfft: int = 4096
    hop: Optional[int] = None
    fs: Optional[float] = None

    def __post_init__(self):
        plt.ion()
        self.fig, (self.ax_psd, self.ax_loss) = plt.subplots(2, 1, figsize=(10, 7))
        self.fig.tight_layout(pad=2.0)

        self.lx, = self.ax_psd.plot([], [], label="x_ref")
        self.ly, = self.ax_psd.plot([], [], label="y_true")
        self.lyh, = self.ax_psd.plot([], [], label="y_hat")
        self.le, = self.ax_psd.plot([], [], label="err = y_hat - y_true")

        self.ax_psd.set_xlabel("Frequency (Hz)" if self.fs is not None else "Frequency (normalized)")
        self.ax_psd.set_ylabel("PSD (dB)")
        self.ax_psd.grid(True)
        self.ax_psd.legend()

        self.train_hist = []
        self.val_hist = []
        self.lt, = self.ax_loss.plot([], [], label="train")
        self.lv, = self.ax_loss.plot([], [], label="val")
        self.ax_loss.set_title("Convergence")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True)
        self.ax_loss.legend()

    def update(
        self,
        x_ref: np.ndarray,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        train_loss: float,
        val_loss: float,
        epoch: int,
    ) -> None:
        f, psd_x = compute_psd_welch(x_ref, nfft=self.nfft, hop=self.hop, fs=self.fs)
        _, psd_y = compute_psd_welch(y_true, nfft=self.nfft, hop=self.hop, fs=self.fs)
        _, psd_yh = compute_psd_welch(y_hat, nfft=self.nfft, hop=self.hop, fs=self.fs)

        psd_x_db = psd_to_db(psd_x)
        psd_y_db = psd_to_db(psd_y)
        psd_yh_db = psd_to_db(psd_yh)

        err = y_hat - y_true
        _, psd_e = compute_psd_welch(err, nfft=self.nfft, hop=self.hop, fs=self.fs)
        psd_e_db = psd_to_db(psd_e)

        psd_x_db_n = normalize_psd_to_0db(psd_x_db, psd_x_db)
        psd_y_db_n = normalize_psd_to_0db(psd_y_db, psd_x_db)
        psd_yh_db_n = normalize_psd_to_0db(psd_yh_db, psd_x_db)
        psd_e_db_n = normalize_psd_to_0db(psd_e_db, psd_x_db)

        self.lx.set_data(f, psd_x_db_n)
        self.ly.set_data(f, psd_y_db_n)
        self.lyh.set_data(f, psd_yh_db_n)
        self.le.set_data(f, psd_e_db_n)
        self.ax_psd.relim()
        self.ax_psd.autoscale_view()

        self.train_hist.append(float(train_loss))
        self.val_hist.append(float(val_loss))
        xs = np.arange(1, len(self.train_hist) + 1)

        self.lt.set_data(xs, self.train_hist)
        self.lv.set_data(xs, self.val_hist)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.ax_psd.set_title(f"Epoch {epoch}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
