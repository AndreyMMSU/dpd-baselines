import os
import numpy as np
import torch
from scipy.io import loadmat

from dpd_baselines.utils.data_loader import BlackBoxData80Loader
from dpd_baselines.models.Hello_world import Hello_world_model
from dpd_baselines.utils.live_monitor import LiveMonitor
from dpd_baselines.signals.filter_design import make_BL

def main():
    mat_path = "data/BlackBoxData_80.mat"
    seq_len = 2**10
    batch_size = 8
    epochs = 1000
    lr = 1e-3
    fs = 1.2288e6
    fc = 0.2e6                              
    numtaps = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    save_path = "checkpoints/first_run.pt"
    os.makedirs("checkpoints", exist_ok=True)

    data = BlackBoxData80Loader(mat_path, seq_len=seq_len)
    data.normalize()
    BL_coeff = data.apply_bl(fc=fc, numtaps=numtaps)

    x_t, y_t, _ = data.get_signals()

    n_train = int(x_t.shape[0])
    x_train, _ = x_t[:n_train], x_t[n_train:]
    y_train, _ = y_t[:n_train], y_t[n_train:]

    monitor = LiveMonitor(nfft=512, fs=fs)
    model = Hello_world_model(filter_order_in=3, filter_order_out=10, poly_order0=5, poly_order1=10, BL_coeff=BL_coeff).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def nmse(y_hat, y_true, ref, eps=1e-12):
        err = (y_hat - y_true).abs().pow(2).mean()
        ref = ref.abs().pow(2).mean()
        return 10*torch.log10(err / (ref + eps))
    
    def mse(y_hat, y_true):
        err = (y_hat - y_true).abs().pow(2).mean()
        return err 
        
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        steps = 0

        for i in range(0, n_train, batch_size):
            xb = x_train[i : i + batch_size].to(device)  
            yb = y_train[i : i + batch_size].to(device)

            optimizer.zero_grad()
            y_hat = model(xb)                
            loss = mse(y_hat, yb)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
            steps += 1

        train_loss = running / max(1, steps)
        model.eval()
        with torch.no_grad():
            y_hat = model(x_train)
            val_loss = float(nmse(y_hat, y_train, ref=x_train).cpu())

            x_ref_r = x_train.reshape(-1).to(device)   
            y_ref_r = y_train.reshape(-1).to(device)   
            y_hat_r = y_hat.reshape(-1).to(device)   

            x_ref_np = x_ref_r.detach().cpu().numpy()
            y_true_np = y_ref_r.detach().cpu().numpy()
            y_hat_np = y_hat_r.detach().cpu().numpy()

        model.train()
        monitor.update(
            x_ref=x_ref_np,
            y_true=y_true_np,
            y_hat=y_hat_np,
            train_loss=float(val_loss),  
            epoch=epoch,
        )

        print(f"epoch {epoch:02d} | mse={train_loss:.3f} | NMSE={val_loss:.3f}")
        

    torch.save({"state_dict": model.state_dict()}, save_path)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
