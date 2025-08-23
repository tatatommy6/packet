# ------------------------------------------------------------
# Interval 기반 CSV를 읽어 10초 시계열을 LSTM 입력 형식으로 변환하고
# train/valid/test로 시간순 분할 + (train에만 fit한) StandardScaler 적용 후
# 모두 .npz 파일로 저장합니다.
#
# 사용법:
#   python preprocess.py --csv packets_10s.csv --out data/prepared.npz
# 옵션:
#   --lookback 12  (지난 12스텝 = 120초)
#   --horizon 1    (다음 1스텝 = 10초)
#   --topk_proto 6 (프로토콜 상위 K개만 피처에 포함)
# ------------------------------------------------------------
import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def make_sequences(X, Y, lookback=12, horizon=1):
    Xs, Ys = [], []
    for i in range(len(X) - lookback - horizon + 1):
        Xs.append(X[i:i+lookback])
        Ys.append(Y[i+lookback + horizon - 1])
    return np.stack(Xs), np.stack(Ys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--topk_proto", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) CSV 로드
    df = pd.read_csv(args.csv).fillna(0.0)

    # 2) 피처/타깃 선정
    base_feats = ["pkt_count","len_mean","len_std","len_max","tcp_ratio","udp_ratio"]
    proto_candidates = ["TCP","UDP","DNS","QUIC","TLSv1.3","TLSv1.2","ICMP","HTTP","MDNS","NTP","OCSP"]

    feature_cols = [c for c in base_feats if c in df.columns]
    present_protos = [p for p in proto_candidates if p in df.columns]
    if present_protos:
        topK = list(df[present_protos].sum().sort_values(ascending=False).index[:args.topk_proto])
        feature_cols += topK

    target_cols = [c for c in ["pkt_count","tcp_ratio","udp_ratio"] if c in df.columns]
    if not target_cols:
        raise ValueError("타깃 컬럼이 없습니다. (pkt_count/tcp_ratio/udp_ratio 중 하나 필요)")

    X_all = df[feature_cols].astype(np.float32).values
    Y_all = df[target_cols].astype(np.float32).values

    # 3) 시퀀스화
    X_seq, Y_seq = make_sequences(X_all, Y_all, args.lookback, args.horizon)
    N = len(X_seq)
    if N <= 50:
        raise ValueError(f"시퀀스 샘플 수가 너무 적습니다. (현재 {N}) 더 오래 수집하거나 lookback/horizon 조정 필요")

    # 4) 시간순 분할
    tr_end = int(N*0.7)
    va_end = int(N*0.85)
    X_tr_raw, Y_tr = X_seq[:tr_end], Y_seq[:tr_end]
    X_va_raw, Y_va = X_seq[tr_end:va_end], Y_seq[tr_end:va_end]
    X_te_raw, Y_te = X_seq[va_end:], Y_seq[va_end:]

    # 5) StandardScaler (train에만 fit)
    B, T, F = X_tr_raw.shape
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw.reshape(-1, F)).reshape(B, T, F)

    def transform_block(X_raw):
        b, t, f = X_raw.shape
        return scaler.transform(X_raw.reshape(-1, f)).reshape(b, t, f)

    X_va = transform_block(X_va_raw)
    X_te = transform_block(X_te_raw)

    # 6) 저장 (.npz)
    np.savez_compressed(
        args.out,
        X_tr=X_tr, Y_tr=Y_tr,
        X_va=X_va, Y_va=Y_va,
        X_te=X_te, Y_te=Y_te,
        feature_cols=np.array(feature_cols, dtype=object),
        target_cols=np.array(target_cols, dtype=object),
        lookback=np.array([args.lookback]),
        horizon=np.array([args.horizon]),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_
    )

    print(f"[OK] Saved -> {args.out}")
    print(f"Samples: train={len(X_tr)}, valid={len(X_va)}, test={len(X_te)}")
    print("Features:", feature_cols)
    print("Targets :", target_cols)

if __name__ == "__main__":
    main()
