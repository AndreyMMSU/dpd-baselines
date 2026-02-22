import os
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy import signal

from dpd_baselines.models.derr_branch_model import FilteredGMP_DN
from dpd_baselines.utils.live_monitor import LiveMonitor


def nmse_db(y_hat: torch.Tensor, y_true: torch.Tensor, ref: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """
    NMSE in dB:
        10 log10( E|y_hat-y_true|^2 / E|ref|^2 )
    """
    err = (y_hat - y_true).abs().pow(2).mean()
    p_ref = ref.abs().pow(2).mean()
    return 10.0 * torch.log10(err / (p_ref + eps))


def _lowpass_filtfilt(z: np.ndarray, fc: float, fs: float, tw: float) -> np.ndarray:
    numtaps = int(np.ceil(4 * fs / tw))
    numtaps |= 1
    b = signal.firwin(numtaps, cutoff=fc, fs=fs, window="hann", pass_zero="lowpass")
    return signal.filtfilt(b, [1.0], z)


def main() -> None:
    # -------------------------
    # config
    # -------------------------
    mat_path = Path("data/BlackBoxData_80.mat")
    x_key = "x"
    y_key = "eRef"  # <-- твоя постановка: desired error = PA_out - X

    seq_len = 2**10
    batch_size = 8
    epochs = 2000
    lr = 1e-3

    # optional LPF (как в твоём примере)
    use_lpf = True
    fc = 0.3e6
    fs = 1.2288e6
    tw = 0.1e6

    # what to show in LiveMonitor: "full" or "z1"
    monitor_mode = "full"   # "z1" if you want only z1 in spectrum plots

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / "run_filtered_gmp_dn.pt"

    # -------------------------
    # load data
    # -------------------------
    m = loadmat(str(mat_path))
    if x_key not in m:
        raise KeyError(f"x_key='{x_key}' not found. Available: {list(m.keys())}")
    if y_key not in m:
        raise KeyError(f"y_key='{y_key}' not found. Available: {list(m.keys())}")

    x = np.asarray(m[x_key]).squeeze()
    y = np.asarray(m[y_key]).squeeze()

    if use_lpf:
        x = _lowpass_filtfilt(x, fc=fc, fs=fs, tw=tw)
        y = _lowpass_filtfilt(y, fc=fc, fs=fs, tw=tw)

    x_t = torch.as_tensor(x.astype(np.complex64))
    y_t = torch.as_tensor(y.astype(np.complex64))

    # normalize by peak (as in your launcher)
    scale = x_t.abs().max().clamp_min(1e-12)
    x_t = x_t / scale
    y_t = y_t / scale

    # reshape into (B, T)
    N = x_t.numel()
    Nw = (N // seq_len)
    x_t = x_t[: Nw * seq_len].view(Nw, seq_len)
    y_t = y_t[: Nw * seq_len].view(Nw, seq_len)

    n_train = int(0.9 * Nw)
    x_train, x_val = x_t[:n_train], x_t[n_train:]
    y_train, y_val = y_t[:n_train], y_t[n_train:]

    # -------------------------
    # model config
    # -------------------------
    # few nonlinear branches (minimal GMP) + per-branch FIR + per-branch complex gain
    orders = [1, 3, 5]
    delays = [0, 1, 2]               # try [0,0,0] if you want no explicit delays
    fir_orders = [7, 7, 7]           # <=0 -> None

    model = FilteredGMP_DN(
        orders=orders,
        delays=delays,
        fir_orders=[m if (m is not None and m > 0) else None for m in fir_orders],
        eps=1e-12,
    ).to(device)

    warmup = model.warmup()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    monitor = LiveMonitor(nfft=512, fs=fs)

    # -------------------------
    # train
    # -------------------------
    model.train()
    for epoch in range(1, epochs + 1):
        running_full = 0.0
        running_z1 = 0.0
        steps = 0

        for i in range(0, n_train, batch_size):
            xb = x_train[i : i + batch_size].to(device)
            yb = y_train[i : i + batch_size].to(device)

            optimizer.zero_grad(set_to_none=True)

            # full output + debug
            y_hat_full, dbg = model.forward_with_debug(xb)
            z1 = dbg["z1"]

            # loss by FULL model (this matches your current behavior)
            loss_full = nmse_db(y_hat_full[:, warmup:], yb[:, warmup:], ref=xb[:, warmup:])
            loss_full.backward()
            optimizer.step()

            # just for logging: NMSE of z1-only
            with torch.no_grad():
                loss_z1 = nmse_db(z1[:, warmup:], yb[:, warmup:], ref=xb[:, warmup:])

            running_full += float(loss_full.detach().cpu())
            running_z1 += float(loss_z1.detach().cpu())
            steps += 1

        train_full = running_full / max(1, steps)
        train_z1 = running_z1 / max(1, steps)

        # -------------------------
        # validation + monitor
        # -------------------------
        model.eval()
        with torch.no_grad():
            xv = x_val.to(device)
            yv = y_val.to(device)

            y_hat_val_full, dbg_v = model.forward_with_debug(xv)
            z1_val = dbg_v["z1"]

            val_full = float(nmse_db(y_hat_val_full[:, warmup:], yv[:, warmup:], ref=xv[:, warmup:]).detach().cpu())
            val_z1 = float(nmse_db(z1_val[:, warmup:], yv[:, warmup:], ref=xv[:, warmup:]).detach().cpu())

            # monitor batch
            x_ref_bt = x_val[:5].to(device)
            y_true_bt = y_val[:5].to(device)

            y_hat_bt_full, dbg_bt = model.forward_with_debug(x_ref_bt)
            z1_bt = dbg_bt["z1"]

            if monitor_mode == "z1":
                y_hat_bt = z1_bt
            else:
                y_hat_bt = y_hat_bt_full

            x_ref_np = x_ref_bt.reshape(-1).detach().cpu().numpy()
            y_true_np = y_true_bt.reshape(-1).detach().cpu().numpy()
            y_hat_np = y_hat_bt.reshape(-1).detach().cpu().numpy()

        model.train()

        monitor.update(
            x_ref=x_ref_np,
            y_true=y_true_np,
            y_hat=y_hat_np,
            train_loss=float(train_full),  # training metric for FULL model
            val_loss=float(val_full),      # validation metric for FULL model
            epoch=epoch,
        )

        g1 = complex(float(model.g1_re.detach().cpu()), float(model.g1_im.detach().cpu()))
        print(
            f"epoch {epoch:04d} | "
            f"train(full)={train_full:.3f} val(full)={val_full:.3f} | "
            f"train(z1)={train_z1:.3f} val(z1)={val_z1:.3f} | "
            f"g1={g1.real:+.3e}{g1.imag:+.3e}j"
        )

    # -------------------------
    # save
    # -------------------------
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "mat_path": str(mat_path),
                "x_key": x_key,
                "y_key": y_key,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "orders": orders,
                "delays": delays,
                "fir_orders": fir_orders,
                "warmup": warmup,
                "use_lpf": use_lpf,
                "fc": fc,
                "fs": fs,
                "tw": tw,
                "monitor_mode": monitor_mode,
            },
        },
        str(save_path),
    )
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
