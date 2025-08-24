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
    def __init__(self, in_dim, hidden, layers, out_dim=3, dropout=0.45, head_dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)  # layers>=2일 때만 적용
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1]
        return self.head(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)           # preprocess.py 결과 .npz
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--head_dropout", type=float, default=0.25)
    ap.add_argument("--ckpt", default="models/lstm.pt")
    args = ap.parse_args()

    import random
    def fix_seed(seed=42):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
    fix_seed(42)


    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

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
# --- 검증 MAE (타깃별) 계산 함수 ---
    def epoch_mae_vec(dl):
        model.eval()
        k = len(target_cols)
        tot = torch.zeros(k)  # CPU 상에서 누적
        n = 0
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(DEVICE)
                pred = model(xb).cpu()          # (B,3)
                tot += torch.abs(pred - yb).sum(dim=0)  # 타깃별 합
                n += len(yb)
        return (tot / max(n, 1)).numpy()        # [pkt, tcp, udp]

    # --- 가중치 (중요도에 맞춰 조절 가능) ---
    w = np.array([1.7, 1.0, 1.0], dtype=np.float32)  # 예: pkt 덜 중요하면 [0.5,1.0,1.0] 등

    best_score = float('inf')   # MAE·w 가중합 기준
    best_ep = -1
    best_va_huber = float('inf')

    for ep in range(1, args.epochs + 1):
        tr_huber = run_epoch(dl_tr, True)          # 학습 손실(Huber, mean)
        va_huber = run_epoch(dl_va, False)         # 검증 손실(Huber, mean)
        va_mae_vec = epoch_mae_vec(dl_va)          # [pkt, tcp, udp] MAE
        va_score   = float((va_mae_vec * w).sum()) # 저장 판단용 점수

        improved = va_score < best_score
        if improved:
            best_score = va_score
            best_ep = ep
            best_va_huber = va_huber
            state = {
                "model": model.state_dict(),
                "epoch": ep,
                "val_huber": va_huber,
                "val_mae_vec": va_mae_vec,
                "val_score": best_score,
                "weights": w,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "lookback": lookback,
                "horizon": horizon,
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale,
            }
            torch.save(state, args.ckpt)
            print(f"[BEST] epoch={ep} val_score(MAE·w)={va_score:.4f} "
                f"vec={np.round(va_mae_vec, 4)}")

        if ep % 5 == 0 or ep == 1:
            print(f"epoch: {ep:02d} train(huber) {tr_huber:.4f} | valid(huber) {va_huber:.4f} "
                f"| valid(MAE·sum) {va_mae_vec.sum():.4f} | best@{best_ep} {best_score:.4f}")

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
    # print("\nSaved:", args.ckpt)
    # print("Features used:", feature_cols)
    # print("Targets:", target_cols)

if __name__ == "__main__":
    main()
