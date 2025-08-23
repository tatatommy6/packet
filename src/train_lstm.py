# train_pytorch.py
# ------------------------------------------------------------
# preprocess.py가 만든 .npz 파일을 읽어 PyTorch LSTM을 학습합니다.
#
# 사용법:
#   python train_lstm.py --data data/prepared.npz --epochs 30
# ------------------------------------------------------------
import argparse, math, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SeqDS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

class LSTMReg(nn.Module):
    def __init__(self, in_dim, hidden, layers, out_dim=3, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, out_dim))
    def forward(self, x):
        out, _ = self.lstm(x)     # (B,T,H)
        h = out[:, -1]            # 마지막 스텝
        return self.head(h)       # (B,out_dim)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)           # preprocess.py 결과 .npz
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ckpt", default="models/lstm_qos.pt")
    args = ap.parse_args()

    DEVICE = "mps" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    # 1) 데이터 로드
    data = np.load(args.data, allow_pickle=True)
    X_tr, Y_tr = data["X_tr"], data["Y_tr"]
    X_va, Y_va = data["X_va"], data["Y_va"]
    X_te, Y_te = data["X_te"], data["Y_te"]
    feature_cols = list(data["feature_cols"])
    target_cols  = list(data["target_cols"])
    lookback     = int(data["lookback"][0])
    horizon      = int(data["horizon"][0])
    scaler_mean  = data["scaler_mean"]
    scaler_scale = data["scaler_scale"]

    dl_tr = DataLoader(SeqDS(X_tr, Y_tr), batch_size=args.batch, shuffle=True, drop_last=True)
    dl_va = DataLoader(SeqDS(X_va, Y_va), batch_size=args.batch, shuffle=False)
    dl_te = DataLoader(SeqDS(X_te, Y_te), batch_size=args.batch, shuffle=False)

    # 2) 모델/학습 준비
    in_dim, out_dim = X_tr.shape[-1], Y_tr.shape[-1]
    model = LSTMReg(in_dim, hidden=args.hidden, layers=args.layers, out_dim=out_dim).to(DEVICE)
    loss_fn = nn.HuberLoss(delta=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def run_epoch(dl, train=True):
        model.train(mode=train)
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.set_grad_enabled(train):
                pred = model(xb)
                loss = loss_fn(pred, yb)
                if train:
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        return tot / max(n,1)

    best_va = math.inf
    for ep in range(1, args.epochs+1):
        tr = run_epoch(dl_tr, True)
        va = run_epoch(dl_va, False)
        if va < best_va:
            best_va = va
            torch.save({
                "model": model.state_dict(),
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "lookback": lookback,
                "horizon": horizon,
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale
            }, args.ckpt)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch: {ep:02d} train {tr:.4f} | valid {va:.4f}")

    # 3) 테스트 평가
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy()); gts.append(yb.numpy())
    preds, gts = np.vstack(preds), np.vstack(gts)
    mae = np.mean(np.abs(preds - gts), axis=0)

    print("\n=== Test MAE per target ===")
    for name, m in zip(target_cols, mae):
        print(f"{name}: {m:.4f}")
    print("\nSaved:", args.ckpt)
    print("Features used:", feature_cols)
    print("Targets:", target_cols)

if __name__ == "__main__":
    main()
