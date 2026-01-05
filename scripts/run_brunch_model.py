import os
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from dpd_baselines.models.branch_model import BranchModel
from dpd_baselines.utils.live_monitor import LiveMonitor


def nmse_db(y_hat: torch.Tensor, y_true: torch.Tensor, ref: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    err = (y_hat - y_true).abs().pow(2).mean()
    p_ref = ref.abs().pow(2).mean()
    return 10.0 * torch.log10(err / (p_ref + eps))


def main() -> None:
    mat_path = Path("data/BlackBoxData_200.mat")
    seq_len = 2**10
    batch_size = 8
    epochs = 500
    lr = 1e-2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / "first_run_branch_model.pt"

    m = loadmat(str(mat_path))
    x = np.asarray(m["x"]).squeeze()
    y = np.asarray(m["y"]).squeeze()

    x_t = torch.as_tensor(x.astype(np.complex64))
    y_t = torch.as_tensor(y.astype(np.complex64))

    scale = x_t.abs().max().clamp_min(1e-12)
    x_t = x_t / scale
    y_t = y_t / scale

    N = x_t.numel()
    Nw = (N // seq_len)
    x_t = x_t[: Nw * seq_len].view(Nw, seq_len)
    y_t = y_t[: Nw * seq_len].view(Nw, seq_len)

    n_train = int(0.9 * Nw)
    x_train, x_val = x_t[:n_train], x_t[n_train:]
    y_train, y_val = y_t[:n_train], y_t[n_train:]

    in_delays = torch.tensor([0, 1, 2], dtype=torch.int64)
    in_fir_orders = torch.tensor([3, 3, 3], dtype=torch.int64)
    in_poly_orders = torch.tensor([2, 2, 2], dtype=torch.int64)

    out_delays = torch.tensor(
        [
            [-1, 0, 1, 2, 3],
            [-1, 0, 1, 2, 3],
            [-1, 0, 1, 2, 3],
        ],
        dtype=torch.int64,
    )
    out_fir_orders = torch.full((3, 5), 3, dtype=torch.int64)
    out_poly_orders = torch.full((3, 5), 3, dtype=torch.int64)

    model = BranchModel(
        in_delays=in_delays,
        in_fir_orders=in_fir_orders,
        in_poly_orders=in_poly_orders,
        out_delays=out_delays,
        out_fir_orders=out_fir_orders,
        out_poly_orders=out_poly_orders,
        poly_init="identity",
        out_fir_order=5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    monitor = LiveMonitor(nfft=512)

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        steps = 0

        for i in range(0, n_train, batch_size):
            xb = x_train[i : i + batch_size].to(device)
            yb = y_train[i : i + batch_size].to(device)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(xb)
            loss = nmse_db(y_hat, yb, ref=xb)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
            steps += 1

        train_loss = running / max(1, steps)

        model.eval()
        with torch.no_grad():
            xv = x_val.to(device)
            yv = y_val.to(device)

            y_hat_val = model(xv)
            val_loss = float(nmse_db(y_hat_val, yv, ref=xv).detach().cpu())

            x_ref_bt = x_val[:3].to(device)
            y_true_bt = y_val[:3].to(device)
            y_hat_ref_bt = model(x_ref_bt)

            x_ref_np = x_ref_bt.reshape(-1).detach().cpu().numpy()
            y_true_np = y_true_bt.reshape(-1).detach().cpu().numpy()
            y_hat_np = y_hat_ref_bt.reshape(-1).detach().cpu().numpy()

        model.train()

        monitor.update(
            x_ref=x_ref_np,
            y_true=y_true_np,
            y_hat=y_hat_np,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            epoch=epoch,
        )

        print(f"epoch {epoch:03d} | train={train_loss:.3f} | val={val_loss:.3f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "in_delays": in_delays.tolist(),
                "in_fir_orders": in_fir_orders.tolist(),
                "in_poly_orders": in_poly_orders.tolist(),
                "out_delays": out_delays.tolist(),
                "out_fir_orders": out_fir_orders.tolist(),
                "out_poly_orders": out_poly_orders.tolist(),
            },
        },
        str(save_path),
    )
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
