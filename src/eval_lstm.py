# src/eval_lstm.py
import argparse, numpy as np, torch, torch.nn as nn

def detect_head_kind(state_dict):
    # head.0.weight가 1D이면 LayerNorm 헤드(학습 때 head[0]=LayerNorm)
    w0 = state_dict.get("head.0.weight", None)
    if isinstance(w0, torch.Tensor) and w0.dim() == 1:
        return "ln"
    # head.5.weight가 있으면 (ln_head: LN, Dropout, Linear, ReLU, Dropout, Linear) 패턴일 확률 큼
    if "head.5.weight" in state_dict:
        return "ln"
    return "basic"

class LSTMReg(nn.Module):
    def __init__(self, in_dim, hidden, layers, out_dim=3,
                lstm_dropout=0.3, head_dropout=0.25, head_kind="basic"):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=lstm_dropout)
        if head_kind == "ln":
            # 학습 시 사용했던 LayerNorm 기반 head와 동일한 모듈 이름: 'head'
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(head_dropout),
                nn.Linear(hidden, 128), nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(128, out_dim),
            )
        else:
            # 기본 head: Linear → ReLU → Linear (모듈 이름은 동일하게 'head')
            self.head = nn.Sequential(
                nn.Linear(hidden, 128), nn.ReLU(),
                nn.Linear(128, out_dim),
            )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1]
        return self.head(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)      # preprocess.py가 만든 .npz
    ap.add_argument("--ckpt", required=True)      # train 시 저장한 .pt
    ap.add_argument("--hidden", type=int, required=True)   # 학습 때 쓴 값 그대로!
    ap.add_argument("--layers", type=int, required=True)   # 학습 때 쓴 값 그대로!
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    # 디바이스
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 데이터 로드
    D = np.load(args.data, allow_pickle=True)
    X_tr, Y_tr = D["X_tr"], D["Y_tr"]
    X_va, Y_va = D["X_va"], D["Y_va"]
    X_te, Y_te = D["X_te"], D["Y_te"]
    feature_cols = list(D["feature_cols"])
    target_cols  = list(D["target_cols"])

    in_dim, out_dim = X_tr.shape[-1], Y_tr.shape[-1]

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt["model"]
    head_kind = detect_head_kind(sd)

    model = LSTMReg(in_dim, args.hidden, args.layers,
                out_dim=out_dim, lstm_dropout=0.3,
                head_dropout=0.25, head_kind=head_kind).to(device)

    model.load_state_dict(sd)   # 이제 키가 정확히 맞습니다
    model.eval()

    # --- 평가 함수들 ---
    def batched_mae(X, Y):
        bs = args.batch
        mae_sum = np.zeros(out_dim, dtype=np.float64); n = 0
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=device)
                yb = torch.tensor(Y[i:i+bs], dtype=torch.float32, device=device)
                pred = model(xb)
                mae_sum += torch.abs(pred - yb).sum(dim=0).cpu().numpy()
                n += len(xb)
        return (mae_sum / max(n,1))

    def last_value_baseline_mae(X, Y):
        # X: (N,T,F), Y: (N,M)
        # target과 동일한 feature를 찾아 마지막 시점 값을 복사하여 예측
        t2idx = {name: feature_cols.index(name) for name in target_cols}
        mae_sum = np.zeros(out_dim, dtype=np.float64); n = 0
        bs = args.batch
        for i in range(0, len(X), bs):
            Xb = X[i:i+bs]; Yb = Y[i:i+bs]
            last = Xb[:, -1, :]  # (B,F)
            pred = np.stack([last[:, t2idx[name]] for name in target_cols], axis=1)  # (B,M)
            mae_sum += np.abs(pred - Yb).sum(axis=0)
            n += len(Xb)
        return (mae_sum / max(n,1))

    # --- 검증/테스트 성능 ---
    va_mae = batched_mae(X_va, Y_va)
    te_mae = batched_mae(X_te, Y_te)
    bl_mae = last_value_baseline_mae(X_te, Y_te)

    def pretty(mae):
        return ", ".join(f"{n}: {v:.4f}" for n, v in zip(target_cols, mae))

# eval_lstm.py 끝부분에 추가(간단 드롭인)

    def batched_preds(X):
        bs = args.batch
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=device)
                out.append(model(xb).cpu().numpy())
        return np.vstack(out)

    def last_value_of(X, feat_name):
        idx = feature_cols.index(feat_name)
        return X[:, -1, idx]

    # 1) 예측/라스트 추출
    pred_va = batched_preds(X_va); pred_te = batched_preds(X_te)
    last_va = last_value_of(X_va, "pkt_count"); last_te = last_value_of(X_te, "pkt_count")
    y_va = Y_va[:, 0]; y_te = Y_te[:, 0]

    # 2) 검증으로 최적 alpha 찾기 (0~1)
    alphas = np.linspace(0, 1, 21)
    mae_list = []
    for a in alphas:
        blend = a*last_va + (1-a)*pred_va[:, 0]
        mae_list.append(np.mean(np.abs(blend - y_va)))
    a_star = alphas[int(np.argmin(mae_list))]
    print(f"[Blend] best alpha on valid = {a_star:.2f}")

    # 3) 테스트에 적용
    blend_te = a_star*last_te + (1-a_star)*pred_te[:, 0]
    pkt_mae_blend = np.mean(np.abs(blend_te - y_te))
    print(f"[Test pkt_count MAE] model={np.mean(np.abs(pred_te[:,0]-y_te)):.2f} | "
        f"last={np.mean(np.abs(last_te-y_te)):.2f} | blend={pkt_mae_blend:.2f}")


    print("\n[Validation MAE] ", pretty(va_mae))
    print("[ Test MAE ]     ", pretty(te_mae))
    print("[ Baseline MAE ] ", pretty(bl_mae))

    # 상대 개선도(%) – 베이스라인 대비 낮을수록 좋음
    improve = 100.0 * (1.0 - te_mae / np.maximum(bl_mae, 1e-12))
    print("[ Improvement vs Baseline (%)] ",
        ", ".join(f"{n}: {g:.1f}%" for n, g in zip(target_cols, improve)))

if __name__ == "__main__":
    main()