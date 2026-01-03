import os
import numpy as np
import torch
from scipy.io import loadmat
from dpd_baselines.models.Hello_world import Hello_world_model

def main():
    mat_path = "data/BlackBoxData_80.mat"
    seq_len = 4096
    batch_size = 64
    epochs = 5
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "checkpoints/first_run.pt"

    os.makedirs("checkpoints", exist_ok=True)

    m = loadmat(mat_path)
    x = np.asarray(m["x"]).squeeze()  
    y = np.asarray(m["y"]).squeeze()  
    x_t = torch.as_tensor(x.astype(np.complex64))  # (N,)
    y_t = torch.as_tensor(y.astype(np.complex64))  # (N,)

    N = x_t.numel()
    Nw = (N - seq_len) // seq_len  # non-overlap windows count
    x_t = x_t[: Nw * seq_len].view(Nw, seq_len)
    y_t = y_t[: Nw * seq_len].view(Nw, seq_len)

    # --- split train/val ---
    n_train = int(0.9 * Nw)
    x_train, x_val = x_t[:n_train], x_t[n_train:]
    y_train, y_val = y_t[:n_train], y_t[n_train:]

    # --- model + optimizer ---
    model = Hello_world_model(filter_order=5, poly_order0=3, poly_order1=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # simple loss: MSE on complex => mean(|e|^2)
    def complex_mse(y_hat, y_true):
        return (y_hat - y_true).abs().pow(2).mean()

    # --- training loop ---
    model.train()
    for epoch in range(1, epochs + 1):
        # shuffle windows
        perm = torch.randperm(n_train)
        x_train = x_train[perm]
        y_train = y_train[perm]

        running = 0.0
        steps = 0

        for i in range(0, n_train, batch_size):
            xb = x_train[i : i + batch_size].to(device)  # (B,T) complex
            yb = y_train[i : i + batch_size].to(device)

            optimizer.zero_grad()
            y_hat = model(xb)                 # (B,T) complex
            loss = complex_mse(y_hat, yb)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
            steps += 1

        train_loss = running / max(1, steps)

        # --- validation ---
        model.eval()
        with torch.no_grad():
            y_hat_val = model(x_val.to(device))
            val_loss = float(complex_mse(y_hat_val, y_val.to(device)).cpu())
        model.train()

        print(f"epoch {epoch:02d} | train={train_loss:.6e} | val={val_loss:.6e}")

    # --- save ---
    torch.save({"state_dict": model.state_dict()}, save_path)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
