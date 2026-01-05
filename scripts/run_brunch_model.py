import os
import numpy as np
import torch
from scipy.io import loadmat
from dpd_baselines.models.branch_model import branch_model
from dpd_baselines.utils.live_monitor import LiveMonitor


def main():
    mat_path = "data/BlackBoxData_200.mat"
    seq_len = 2**10
    batch_size = 8
    epochs = 500
    lr = 1e-4
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

    monitor = LiveMonitor(nfft=512)
    model = branch_model(filter_order=10, poly_order0=5, poly_order1=5, poly_order2=5, poly_order3=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def nmse(y_hat, y_true, ref, eps=1e-12):
        err = (y_hat - y_true).abs().pow(2).mean()
        ref = ref.abs().pow(2).mean()
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
            xv = x_val.to(device)
            yv = y_val.to(device)
            y_hat_val = model(xv)
            val_loss = float(nmse(y_hat_val, yv, ref=xv).cpu())

            x_ref_t = x_val[:3].reshape(-1).to(device)   
            y_ref_t = y_val[:3].reshape(-1).to(device)   

            x_ref_bt = x_val[:3].to(device)              
            y_hat_ref_bt = model(x_ref_bt)               

            x_ref_np = x_ref_t.detach().cpu().numpy()
            y_true_np = y_ref_t.detach().cpu().numpy()
            y_hat_np = y_hat_ref_bt.reshape(-1).detach().cpu().numpy()

        model.train()
        monitor.update(
            x_ref=x_ref_np,
            y_true=y_true_np,
            y_hat=y_hat_np,
            train_loss=float(loss),  
            val_loss=float(val_loss),
            epoch=epoch,
        )

        print(f"epoch {epoch:02d} | train={train_loss:.3f} | val={val_loss:.3f}")
        

    torch.save({"state_dict": model.state_dict()}, save_path)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
