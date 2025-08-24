# ------------------------------------------------------------
# Interval 기반 CSV를 읽어 10초 시계열을 LSTM 입력 형식으로 변환하고
# train/valid/test로 시간순 분할 + (train에만 fit한) StandardScaler 적용 후
# 모두 .npz 파일로 저장함
#
# 사용법(예시):
#   python src/preprocessing.py --csv data/packetsby10s_59m29s.csv --out data/prepared.npz
# 옵션:
#   --lookback 12  (지난 12스텝 = 120초)
#   --horizon 1    (다음 1스텝 = 10초)
#   --topk_proto 6 (프로토콜 상위 K개만 피처에 포함)
# ------------------------------------------------------------
import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# make_seqences() 에서  사용한 알고리즘은 슬라이딩 윈도우 방식(찾아보면 나옴)
# 예시: L(len(X))=10, lookback = 3, horizon = 2 라고 가정했을때
# 길이가 10개라 시간 인덱스는 10개: t0 t1 t2 t3 t4 t5 t6 t7 t8 t9
# 그러면 루프는 10 - 3 - 2 + 1 = 6번(i = 0~5)
# i=0 -> X: t0,t1,t2 / Y: t(0+3+2-1) = t4
# i=1 -> X:    t1,t2,t3 / Y: t(1+3+2-1) = t5
# i=2 -> X:       t2,t3,t4 / Y: t(2+3+2-1) = t6
# 이렇게 돌아간다~
def make_sequences(X, Y, lookback=12, horizon=1):
    Xseq, Yseq = [], []
    for i in range(len(X) - lookback - horizon + 1): # N
        Xseq.append(X[i:i+lookback]) #입력 시퀀스는 i에서 시작해 lookback만큼의 과거 구간 
        Yseq.append(Y[i+lookback + horizon - 1]) #정답은 그 창의 끝 이후로 horizon-1 만큼 이동한 시점의 타깃 1개.
    return np.stack(Xseq), np.stack(Yseq)

def main(): #터미널에서 쓸 argument 정의
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--topk_proto", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True) #출력 폴더가 없으면 생성

    # CSV 로드 (--csv 다음 나오는 .csv 파일이 들어감)
    df = pd.read_csv(args.csv).fillna(0)

    # 피처/타깃 선정
    base_feats = ["pkt_count","len_mean","len_std","len_max","tcp_ratio","udp_ratio"]
    protocol_candidates = ["TCP","UDP","DNS","QUIC","TLSv1.3","TLSv1.2","ICMP","HTTP","MDNS","NTP","OCSP"]

    feature_cols = [c for c in base_feats if c in df.columns] #base_feats에 있는 컬럼 중 csv에 있는 컬럼만 선택
    present_protos = [p for p in protocol_candidates if p in df.columns] #protocol_candidates에 있는 컬럼 중 csv에 있는 컬럼만 선택

    if present_protos: #각 컬럼 별 데이터를 다 더한 후 내림차순 정렬하여 상위 K개 선택(k의 기본값은 2)
        topK = list(df[present_protos].sum().sort_values(ascending=False).index[:args.topk_proto])
        feature_cols += topK 
        #feature_cols = ["pkt_count","len_mean","tcp_ratio"...] + ["TCP","UDP"] 이런식으로 저장됨 

    target_cols = [c for c in ["pkt_count","tcp_ratio","udp_ratio"] if c in df.columns]
    if not target_cols:
        raise ValueError("타깃 컬럼이 없습니다. (pkt_count/tcp_ratio/udp_ratio 중 하나 필요)")

    # input:모델이 보고 학습할 데이터(입력) answer:모델이 맞춰야할 데이터(정답)
    input_all = df[feature_cols].astype(np.float32).values # feature_cols에 있는 컬럼들만 뽑아서 numpy array로 변환
    answer_all = df[target_cols].astype(np.float32).values   # target_cols에 있는 컬럼들만 뽑아서 numpy array로 변환

    # 시퀀스화
    input_seq, answer_seq = make_sequences(input_all, answer_all, args.lookback, args.horizon)
    N = len(input_seq)
    if N <= 50:
        raise ValueError(f"시퀀스 샘플 수가 너무 적습니다. (현재 {N}) 더 오래 수집하거나 lookback/horizon 조정 필요")

# 결과 모양(Shape):
#  input_seq = (샘플 수 N, 입력 길이 lookback, 입력 특징 수 F)
#  answer_seq = (샘플 수 N, 출력(정답) 특징 수 M)
# 몇 개의 샘플(N)을 만들 수 있는지: N = 전체 길이 L - 입력 길이 lookback - horizon + 1

    # 시간순 분할
    train_end = int(N*0.7)  # 훈련데이터
    valid_end = int(N*0.85) # 검증데이터

    input_train_raw, answer_train = input_seq[:train_end], answer_seq[:train_end]
    input_valid_raw, answer_valid = input_seq[train_end:valid_end], answer_seq[train_end:valid_end]
    input_test_raw, answer_test = input_seq[valid_end:], answer_seq[valid_end:]

    # StandardScaler (train에만 fit)
    B, T, F = input_train_raw.shape
    input_train = StandardScaler().fit_transform(input_train_raw.reshape(-1, F)).reshape(B, T, F)

# b (batch size)
# 샘플 개수 (시퀀스 개수, 즉 N)
# 한 번에 몇 개의 시퀀스를 담고 있는지

# t (time steps, lookback 길이)
# 시퀀스 안에서의 시간축 길이
# 몇 스텝 동안 과거 데이터를 입력으로 줄 건지 (예: lookback=12면 12)

# f (features, 입력 변수 개수)
# 각 시점마다 몇 개의 피처(변수)가 있는지
# 예: pkt_count, len_mean, TCP, UDP … 같은 입력 컬럼 수
# ex) 
# X_raw.shape = (1000, 12, 8) 이라면 
# b=1000, t=12, f=8

    def transform_block(input_raw):
        b, t, f = input_raw.shape
        return StandardScaler().transform(input_raw.reshape(-1, f)).reshape(b, t, f)

    input_valid = transform_block(input_valid_raw)
    input_test = transform_block(input_test_raw)

    # 저장 (.npz)
    np.savez_compressed(
        args.out,
        X_tr = input_train, Y_tr = answer_train,
        X_va = input_valid, Y_va = answer_valid,
        X_te = input_test, Y_te = answer_test,
        feature_cols=np.array(feature_cols, dtype=object),
        target_cols=np.array(target_cols, dtype=object),
        lookback=np.array([args.lookback]),
        horizon=np.array([args.horizon]),
        scaler_mean=StandardScaler().mean_,
        scaler_scale=StandardScaler().scale_
    )

    print(f"Saved -> {args.out}")
    print(f"Samples: train={len(input_train)}, valid={len(input_valid)}, test={len(input_test)}")
    print("Features:", feature_cols)
    print("Targets :", target_cols)

if __name__ == "__main__":
    main()