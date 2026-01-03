import os
import numpy as np
import torch
from scipy.io import loadmat
from dpd_baselines.models.Hello_world import Hello_world_model
from dpd_baselines.utils.live_monitor import live_monitor


def main():
    mat_path = "data/BlackBoxData_80.mat"
    seq_len = 2**10
    batch_size = 8
    epochs = 100
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "checkpoints/first_run.pt"

    os.makedirs("checkpoints", exist_ok=True)

    m = loadmat(mat_path)
    x = np.asarray(m["x"]).squeeze()  
    y = np.asarray(m["y"]).squeeze()  
    x_t = torch.as_tensor(x.astype(np.complex64)) 
    y_t = torch.as_tensor(y.astype(np.complex64))  
    y_t = y_t/x_t.abs().max()
    x_t = x_t/x_t.abs().max()

    N = x_t.numel()
    Nw = (N - seq_len) // seq_len  
    x_t = x_t[: Nw * seq_len].view(Nw, seq_len)
    y_t = y_t[: Nw * seq_len].view(Nw, seq_len)

    n_train = int(0.9 * Nw)
    x_train, x_val = x_t[:n_train], x_t[n_train:]
    y_train, y_val = y_t[:n_train], y_t[n_train:]

    model = Hello_world_model(filter_order=5, poly_order0=6, poly_order1=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def nmse(y_hat, y_true, x_ref, eps=1e-12):
        err = (y_hat - y_true).abs().pow(2).mean()
        ref = x_ref.abs().pow(2).mean()
        return 10*torch.log10(err / (ref + eps))
    
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        steps = 0

        for i in range(0, n_train, batch_size):
            xb = x_train[i : i + batch_size].to(device)  
            yb = y_train[i : i + batch_size].to(device)

            optimizer.zero_grad()
            y_hat = model(xb)                
            loss = nmse(y_hat, yb, xb)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
            steps += 1

        train_loss = running / max(1, steps)
        model.eval()
        with torch.no_grad():
            y_hat_val = model(x_val.to(device))
            val_loss = float(nmse(y_hat_val, y_val.to(device), x_val).cpu())
        model.train()

        print(f"epoch {epoch:02d} | train={train_loss:.3f} | val={val_loss:.3f}")

    torch.save({"state_dict": model.state_dict()}, save_path)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
