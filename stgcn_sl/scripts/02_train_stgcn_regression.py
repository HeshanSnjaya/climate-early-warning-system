import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset: sliding windows
# -----------------------------
class SlidingWindowDataset(Dataset):
    def __init__(self, V, P=30, H=17, target_col_idx=None):
        """
        V: numpy array [T, N] (already transformed, e.g., log1p)
        P: past window length
        H: horizon (days ahead)
        target_col_idx: which node(s) to predict; None => predict all N
        """
        self.V = V.astype(np.float32)
        self.P = P
        self.H = H
        self.T, self.N = self.V.shape

        self.start = P
        self.end = self.T - H  # last t such that t+H exists
        self.idxs = np.arange(self.start, self.end, dtype=int)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]
        x = self.V[t - self.P:t, :]          # [P, N]
        y = self.V[t + self.H, :]            # [N]
        return torch.from_numpy(x), torch.from_numpy(y), t


# -----------------------------
# Graph Convolution (simple)
# -----------------------------
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A_norm):
        super().__init__()
        self.A_norm = A_norm  # [N, N] torch tensor
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, X):
        """
        X: [B, N, C_in]
        return: [B, N, C_out]
        """
        # Aggregate neighbors: A_norm @ X
        X_agg = torch.einsum("nm,bmc->bnc", self.A_norm, X)
        return self.lin(X_agg)


# -----------------------------
# STGCN-style block (Time -> Graph -> Time)
# Using 1D temporal conv across time dimension
# -----------------------------
class STBlock(nn.Module):
    def __init__(self, N, Cin, Cmid, Cout, A_norm, Kt=3, dropout=0.1):
        super().__init__()
        self.N = N

        # Temporal conv 1: operates on [B, Cin, P, N]
        self.tconv1 = nn.Conv2d(Cin, Cmid, kernel_size=(Kt, 1), padding=(Kt-1, 0))

        # Graph conv: operates on node dimension after reducing time
        self.gconv = GraphConv(Cmid, Cmid, A_norm)

        # Temporal conv 2
        self.tconv2 = nn.Conv2d(Cmid, Cout, kernel_size=(Kt, 1), padding=(Kt-1, 0))

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, X):
        """
        X: [B, Cin, P, N]
        """
        # Temporal conv 1
        x = self.tconv1(X)        # [B, Cmid, P', N] (P' same due to padding)
        x = self.act(x)
        x = self.dropout(x)

        # Graph conv: apply at each time slice independently
        # reshape to [B*P', N, Cmid] for graph conv
        B, Cmid, Pp, N = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()      # [B, P', N, Cmid]
        x_flat = x_perm.view(B * Pp, N, Cmid)            # [B*P', N, Cmid]

        x_g = self.gconv(x_flat)                         # [B*P', N, Cmid]
        x_g = self.act(x_g)

        # back to [B, Cmid, P', N]
        x_g = x_g.view(B, Pp, N, Cmid).permute(0, 3, 1, 2).contiguous()
        x_g = self.dropout(x_g)

        # Temporal conv 2
        out = self.tconv2(x_g)      # [B, Cout, P', N]
        out = self.act(out)
        out = self.dropout(out)
        return out


# -----------------------------
# Simple STGCN Regression Model
# -----------------------------
class STGCNRegressor(nn.Module):
    def __init__(self, N, A_norm, P=30, Kt=3, hidden1=32, hidden2=32, dropout=0.1):
        super().__init__()
        self.N = N
        self.P = P

        # Input has 1 channel (precip only)
        self.block1 = STBlock(N=N, Cin=1, Cmid=hidden1, Cout=hidden1, A_norm=A_norm, Kt=Kt, dropout=dropout)
        self.block2 = STBlock(N=N, Cin=hidden1, Cmid=hidden2, Cout=hidden2, A_norm=A_norm, Kt=Kt, dropout=dropout)

        # Final: take last time slice and map hidden->1 per node
        self.out_lin = nn.Linear(hidden2, 1)

    def forward(self, x):
        """
        x: [B, P, N]  (precip history)
        return: [B, N]
        """
        x = x.unsqueeze(1)              # [B, 1, P, N]
        x = self.block1(x)              # [B, H1, P, N]
        x = self.block2(x)              # [B, H2, P, N]

        # take last time step features
        x_last = x[:, :, -1, :].permute(0, 2, 1).contiguous()   # [B, N, H2]
        y = self.out_lin(x_last).squeeze(-1)                    # [B, N]
        return y


def normalize_adjacency(W):
    """
    Symmetric normalization: A_hat = D^{-1/2} W D^{-1/2}
    """
    W = W.astype(np.float32)
    d = W.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[d == 0] = 0.0
    D_inv = np.diag(d_inv_sqrt)
    A_hat = D_inv @ W @ D_inv
    return A_hat.astype(np.float32)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    loss_fn = nn.L1Loss()  # MAE
    for x, y, _t in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys, ps, ts = [], [], []
    for x, y, t in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
        ts.append(t.numpy())
    Y = np.concatenate(ys, axis=0)
    P = np.concatenate(ps, axis=0)
    T = np.concatenate(ts, axis=0)

    mae = mean_absolute_error(Y.flatten(), P.flatten())
    rmse = math.sqrt(mean_squared_error(Y.flatten(), P.flatten()))
    return mae, rmse, Y, P, T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--P", type=int, default=30, help="Past window length (days).")
    parser.add_argument("--H", type=int, default=17, help="Forecast horizon (days).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    proc_dir = project_root / "data" / "processed"
    out_dir = project_root / "outputs"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    meta = json.loads((proc_dir / "meta.json").read_text(encoding="utf-8"))
    V_path = proc_dir / meta["V_file"]
    W_path = proc_dir / meta["W_file"]
    stations_path = proc_dir / meta["stations_file"]

    V_df = pd.read_csv(V_path)  # first column is date index
    dates = pd.to_datetime(V_df.iloc[:, 0]).dt.date
    V = V_df.iloc[:, 1:].to_numpy(dtype=np.float32)  # [T, N]
    T_total, N = V.shape

    W = pd.read_csv(W_path, header=None).to_numpy(dtype=np.float32)
    A_hat = normalize_adjacency(W)
    A_hat_t = torch.from_numpy(A_hat)

    # Time split (70/15/15)
    n_train = int(T_total * 0.70)
    n_val = int(T_total * 0.15)
    n_test = T_total - n_train - n_val

    V_train = V[:n_train, :]
    V_val = V[n_train - args.P - args.H: n_train + n_val, :]   # allow windows at boundary
    V_test = V[n_train + n_val - args.P - args.H:, :]

    train_ds = SlidingWindowDataset(V_train, P=args.P, H=args.H)
    val_ds = SlidingWindowDataset(V_val, P=args.P, H=args.H)
    test_ds = SlidingWindowDataset(V_test, P=args.P, H=args.H)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = STGCNRegressor(N=N, A_norm=A_hat_t.to(device), P=args.P, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e9
    best_path = out_dir / "checkpoints" / f"stgcn_precip_P{args.P}_H{args.H}.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mae, val_rmse, *_ = eval_model(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_L1={tr_loss:.4f} | val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), best_path)

    # Load best and test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_mae, test_rmse, Y, P, t_idx = eval_model(model, test_loader, device)

    metrics = {
        "P": args.P,
        "H": args.H,
        "test_MAE": float(test_mae),
        "test_RMSE": float(test_rmse),
        "N_stations": int(N),
        "T_total": int(T_total),
        "V_file": meta["V_file"],
        "W_file": meta["W_file"],
        "transform": meta.get("transform", ""),
    }

    (out_dir / "metrics" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Test metrics:", metrics)

    # Save predictions in long format for dashboard
    stations = pd.read_csv(stations_path)
    test_start_global = n_train + n_val - args.P - args.H
    dates_test = np.array(dates.iloc[test_start_global:])  # dates aligned with V_test rows

    # t_idx are indices in V_test time axis corresponding to "t"
    origin_dates = dates_test[t_idx]
    target_dates = dates_test[t_idx + args.H]

    # Save long format
    rows = []
    for sample_i in range(Y.shape[0]):
        for node in range(N):
            rows.append({
                "origin_date": str(origin_dates[sample_i]),
                "target_date": str(target_dates[sample_i]),
                "station_idx": int(node),
                "latitude": float(stations.loc[stations.station_idx == node, "latitude"].values[0]),
                "longitude": float(stations.loc[stations.station_idx == node, "longitude"].values[0]),
                "y_true": float(Y[sample_i, node]),
                "y_pred": float(P[sample_i, node]),
                "abs_error": float(abs(Y[sample_i, node] - P[sample_i, node])),
            })

    pred_df = pd.DataFrame(rows)
    pred_path = out_dir / "predictions" / f"predictions_P{args.P}_H{args.H}.csv"
    pred_df.to_csv(pred_path, index=False)
    print("Saved predictions:", pred_path)


if __name__ == "__main__":
    import math
    main()
