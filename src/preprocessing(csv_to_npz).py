# ------------------------------------------------------------
# Interval 기반 CSV를 읽어 10초 시계열을 LSTM 입력 형식으로 변환하고
# train/valid/test로 시간순 분할 + (train에만 fit한) StandardScaler 적용 후
# 모두 .npz 파일로 저장함
#
# 사용법(예시):
#   python src/preprocessing(csv_to_npz).py --csv data/packetsby10s_59m29s.csv --out data/prepared.npz
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
def make_sequences(X, Y, lookback=24, horizon=1):
    Xseq, Yseq = [], []
    for i in range(len(X) - lookback - horizon + 1): # N
        Xseq.append(X[i:i+lookback]) # 입력 시퀀스는 i에서 시작해 lookback만큼의 과거 구간 
        Yseq.append(Y[i+lookback + horizon - 1]) # 정답은 그 창의 끝 이후로 horizon-1 만큼 이동한 시점의 타깃 1개.
    return np.stack(Xseq), np.stack(Yseq)

def main(): # 터미널에서 쓸 argument 정의
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lookback", type=int, default=24) # default 부분들을 바꾸면서 학습 시도하자
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--topk_proto", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True) #출력 폴더가 없으면 생성

    # CSV 로드 (--csv 다음 나오는 .csv 파일이 들어감)
    df = pd.read_csv(args.csv).fillna(0)

    # 피처/타깃 선정
    base_feats = ["pkt_count","len_mean","len_std","len_max","tcp_ratio","udp_ratio"]
    protocol_candidates = ["TCP","UDP","DNS","QUIC","TLSv1.3","TLSv1.2","ICMP","HTTP","MDNS","NTP","OCSP"]

    feature_cols = [c for c in base_feats if c in df.columns] # base_feats에 있는 컬럼 중 csv에 있는 컬럼만 선택
    present_protos = [p for p in protocol_candidates if p in df.columns] # protocol_candidates에 있는 컬럼 중 csv에 있는 컬럼만 선택

    if present_protos: # 각 컬럼 별 데이터를 다 더한 후 내림차순 정렬하여 상위 K개 선택(k의 기본값은 2)
        topK = list(df[present_protos].sum().sort_values(ascending=False).index[:args.topk_proto])
        feature_cols += topK 
        # feature_cols = ["pkt_count","len_mean","tcp_ratio"...] + ["TCP","UDP"] 이런식으로 저장됨 

    target_cols = [c for c in ["pkt_count","tcp_ratio","udp_ratio"] if c in df.columns]
    if not target_cols:
        raise ValueError("타깃 컬럼이 없습니다. (pkt_count/tcp_ratio/udp_ratio 중 하나 필요)")

    # input:모델이 보고 학습할 데이터(입력) answer:모델이 맞춰야할 데이터(정답)
    input_all = df[feature_cols].astype(np.float32).values # feature_cols에 있는 컬럼들만 뽑아서 numpy array로 변환
    answer_all = df[target_cols].astype(np.float32).values   # target_cols에 있는 컬럼들만 뽑아서 numpy array로 변환

    # 결과 모양(Shape):
    # input_seq = (샘플 수 N, 입력 길이 lookback, 입력 특징 수 F)
    # answer_seq = (샘플 수 N, 출력(정답) 특징 수 M)
    # 몇 개의 샘플(N)을 만들 수 있는지: N = 전체 길이 L - 입력 길이 lookback - horizon + 1

                            #위에 있는 함수 참고
    input_seq, answer_seq = make_sequences(input_all, answer_all, args.lookback, args.horizon)
    N = len(input_seq)
    if N <= 50:
        raise ValueError(f"시퀀스 샘플 수가 너무 적습니다. (현재 {N}) 더 오래 수집하거나 lookback/horizon 조정 필요")

    # 시간순 분할
    train_end = int(N*0.7)
    valid_end = int(N*0.85)
    # 시간순으로 훈련, 검증, 테스트 데이터 나누는 코드(인공지능코드 찍먹이라도 해봤다면 알겠죠?)
    input_train_raw, answer_train = input_seq[:train_end], answer_seq[:train_end]
    input_valid_raw, answer_valid = input_seq[train_end:valid_end], answer_seq[train_end:valid_end]
    input_test_raw,  answer_test  = input_seq[valid_end:], answer_seq[valid_end:]

    # ---------- StandardScaler: train에만 fit, 같은 스케일러로 모두 transform ----------
    B, T, F = input_train_raw.shape
    scaler = StandardScaler()
    scaler.fit(input_train_raw.reshape(-1, F))  # train의 분포로만 학습

    input_train = scaler.transform(input_train_raw.reshape(-1, F)).reshape(B, T, F).astype(np.float32)

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

    def transform_block_with(same_scaler, input_raw):
            b, t, f = input_raw.shape
            scaled = same_scaler.transform(input_raw.reshape(-1, f)).reshape(b, t, f)
            return scaled.astype(np.float32)

    input_valid = transform_block_with(scaler, input_valid_raw)
    input_test  = transform_block_with(scaler, input_test_raw)

    # 현재까지 만들어진 것을 .npz 파일로 저장하는 코드
    # .npz란? 배열 1개를 numpy format 으로 저장 -> .npy
    # 배열 여러개를 numpy format 으로 저장 -> .npz
    # 근데 여기서 압축하고싶다? -> np.savez_compressed()
    # .npz로 포장하는 이유 -> lstm이 먹기 편하게 가공한 패키지
    # 그냥 csv로 학습을 돌려도 되지만 지금 csv만 0.7GB라 너무 커서 안됨(데이터 누수, 속도 저하 이슈)
    np.savez_compressed(
        # 상기하자! X는 모델이 보고 학습할 데이터(입력) Y는 모델이 맞춰야할 데이터(정답)
        args.out, # .npz 파일을 어디에 저장할지
        X_tr = input_train, Y_tr = answer_train,
        X_va = input_valid, Y_va = answer_valid,
        X_te = input_test,  Y_te = answer_test,
        feature_cols=np.array(feature_cols, dtype=object), # 입력 피처 이름 목록 예: ["pkt_count","len_mean","...","TCP","UDP","DNS", ...]
        target_cols=np.array(target_cols, dtype=object), # 타깃 이름 목록. 예: ["pkt_count","tcp_ratio","udp_ratio"] (CSV에 없는 건 빠질 수 있음)
        lookback=np.array([args.lookback]),
        horizon=np.array([args.horizon]),
        scaler_mean=scaler.mean_.astype(np.float32), # 입력 피처용 StandardScaler 평균(학습셋으로만 fit).
        scaler_scale=scaler.scale_.astype(np.float32) # 입력 피처용 StandardScaler 표준편차(학습셋으로만 fit).
    )
    # 설명 안해도 알거라고 믿습니다.
    print(f"Saved -> {args.out}")
    print(f"Samples: train={len(input_train)}, valid={len(input_valid)}, test={len(input_test)}")
    print("Features:", feature_cols)
    print("Targets :", target_cols)

#이것도요
if __name__ == "__main__":
    main()