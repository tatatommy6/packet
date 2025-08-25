# ------------------------------------------------------------
# preprocess.py가 만든 .npz 파일을 읽어 PyTorch LSTM을 학습합니다.
#
# 사용법:
#   python train_lstm.py --data data/prepared.npz --epochs 30
# 옵션:
#   --hidden: 은닉층 개수 지정(64 ~ 512)
#   --dropout: 드롭아웃 지정(0.2 ~ 0.4)
#   --lr: 학습률 지정(1e-1 ~ 1e-4)
# ------------------------------------------------------------
import random
import argparse, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 입력된 시계열 데이터(X)와 타깃(Y)을 DataLoader가 읽을 수 있게 포장하는 클래스
class SeqDS(Dataset): # torch.utils.data.Dataset를 상속한 커스텀 데이터셋 정의.
    def __init__(self, input, answers):
        #넘파이 input, answers를 torch 텐서로 변환
        # dtype = torch.float32 로 명시적으로 지정(딥러닝에서 효율이 좋음)
        self.input = torch.tensor(input, dtype=torch.float32)
        self.answers = torch.tensor(answers, dtype=torch.float32)

    def __len__(self): return len(self.input) # 데이터셋 길이 반환. DataLoader가 배치 개수 등을 계산할 때 사용. 여기서 return 값은 N(샘플) 수 
    def __getitem__(self, i): return self.input[i], self.answers[i] # 인덱스 i에 해당하는 (입력(X), 정답(Y)) 쌍 반환. DataLoader가 배치 생성 시 사용.

# 이 부분은 저번에 주식이랑 torch.nn라이브러리 찍먹할 때 한번 설명 한적이 있어요
# 그래도 다시 합시다

# LSTM 모델 정의
class LSTMDef(nn.Module):
    def __init__(self, in_dim, hidden, layers, out_dim=3, dropout=0.45, head_dropout=0.25):
        super().__init__() # 부모 클래스의 __init__()을 실행시켜서, 상속받은 기능을 제대로 초기화해주는 코드
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)  # layers >= 2일 때만 적용
        self.head = nn.Sequential( # 여러 레이어를 순서대로 연결해서 모델을 간단하게 만드는 컨테이너
            nn.LayerNorm(hidden),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, out_dim),
        )

    #LSTM 출력 -> 마지막 시점 벡터 뽑기 -> 최종 예측층 통과
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1]
        return self.head(h)

def main(): 
    #전처리 코드에 있는것과 똑같은 어규먼트 설정 코드
    #직관적으로 보임. 굳이 주석 안달아도 알 수 있죠?
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--head_dropout", type=float, default=0.25)
    ap.add_argument("--ckpt", default="models/lstm.pt") #파일 저장 경로 설정 어규먼트
    args = ap.parse_args()

    # 새롭게 알게 된 사실: 한줄이 아닌 코드를 한줄로 쓸때는 세미콜론 필수
    # 아래 함수에 보면 시드 고정을 위해 파이썬 기본 random, numpy random, torch random 다 쓰는데
    # 이유는 학습때 쓰이는 랜덤 소스가 어려가지이기 때문임
    def fix_seed(seed):
        random.seed(seed) # python 기본 RNG(난수생성기)는 데이터 섞기나 샘플링에 쓰임
        np.random.seed(seed) # numpy RNG는 전처리·수치 계산 단계에서 종종 사용
        torch.manual_seed(seed) # PyTorch의 기본 RNG는 (주로 CPU, MPS 포함)에 시드 설정
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed) # 이건 CUDA(GPU) RNG 전체에 시드 설정(지금은 mps쓰니까 상관 없음)
        torch.use_deterministic_algorithms(True)
    fix_seed(42)


    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    print("Using device:", DEVICE)

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    # 1) 데이터 로드
    data = np.load(args.data, allow_pickle=True) # pickle이란: 텍스트 상태의 데이터가 아닌 파이썬 객체 자체를 '파일'로 저장하는 것

    # 딕셔너리 형태로 저장된 데이터에서 키를 이용해 값(데이터셋) 추출해서 변수에 담음
    X_tr, Y_tr = data["X_tr"], data["Y_tr"]
    X_va, Y_va = data["X_va"], data["Y_va"]
    X_te, Y_te = data["X_te"], data["Y_te"]
    feature_cols = list(data["feature_cols"])
    target_cols = list(data["target_cols"])
    lookback = int(data["lookback"][0])
    horizon = int(data["horizon"][0])
    scaler_mean = data["scaler_mean"]
    scaler_scale = data["scaler_scale"]

    dataloader_train = DataLoader(SeqDS(X_tr, Y_tr), batch_size=args.batch, shuffle=True, drop_last=True)
    dataloader_valid = DataLoader(SeqDS(X_va, Y_va), batch_size=args.batch, shuffle=False)
    dl_te = DataLoader(SeqDS(X_te, Y_te), batch_size=args.batch, shuffle=False)

    # 2) 모델/학습 준비
    in_dim, out_dim = X_tr.shape[-1], Y_tr.shape[-1]
    model = LSTMDef(in_dim, 
                    hidden = args.hidden, 
                    layers = args.layers, 
                    out_dim = out_dim).to(DEVICE)
    
    # huber loss: MAE와 MSE의 장점을 결합한 '손실 함수'
    # 오차가 적을땐 MSE처럼 작동하여 민감하게 반응하고
    # 오차가 많을땐 MAE처럼 작동하여 이상치에 둔감함
    # 데이터에 이상치가 존재하는 경우에 자주 사용됨
    loss_func = nn.HuberLoss(delta=1.0)

    # adam 옵티마이저: 딥러닝에서 가장 널리 쓰이는 최적화 알고리즘
    # RMSProp과 모멘텀(Momentum) 최적화 기법의 장점을 결합한 방식
    # 지수이동평균으로 누적된 기울기를 보고 판단하여 가중치를 줄이거나 늘리는 등 동적으로 조절함.
    # 복잡한 모델/데이터, 초기 탐색 속도가 중요, 희소/노이즈 그라디언트 → Adam이 편하고 빠름
    # SGD+Momentum: 이미지 분류 등 대형 비전 과제에서 최종 일반화가 더 좋은 경우가 많음.
    # weight_decay: overfitting을 방지하기 위해
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-4)

    # 한 에폭동안의 평균 손실 값을 return함
    def run_epoch(dataloader, train = True):
        model.train(mode = train)
        total, n = 0.0, 0
        for input_batch, answer_batch in dataloader:
            input_batch, answer_batch = input_batch.to(DEVICE), answer_batch.to(DEVICE)
            with torch.set_grad_enabled(train): # 학습 모드일때만 그래디언트 계산 활성화
                pred = model(input_batch) # lstm모델에 입력을 전달해 예측값 생성
                loss = loss_func(pred, answer_batch) # 손실함수 계산
                if train: # 학습 모드일 경우
                    opt.zero_grad(); loss.backward() # 옵티마이저의 그래디언트를 초기화하고 / 손실의 그래디언트를 계산함.
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip_grad_norm(): 그래디언트 클리핑을 통해 그래디언트 폭파를 막음
                    opt.step() # 가중치 업데이트
            total += loss.item() * len(input_batch) # 손실 값 누적
            n += len(input_batch) # 샘플 개수 증가
        return total / max(n,1)

# 타깃별 MAE를 담은 numpy array return
# 예시: [pkt, tcp, udp]
    def epoch_mae_vec(dataloader):
        model.eval() # 평가 모드로 설정
        k = len(target_cols) # topk(타깃)의 개수
        total = torch.zeros(k) # 타깃별 mae를 누적할 텐서 초기화
        n = 0
        with torch.no_grad():
            for input_batch, output_batch in dataloader:
                input_batch = input_batch.to(DEVICE)
                pred = model(input_batch).cpu()
                total += torch.abs(pred - output_batch).sum(dim=0)  # 예측값과 정답의 절대오차를 계산하고 타깃별로 합산
                n += len(output_batch) 
        return (total / max(n, 1)).numpy()

    # --- 가중치 (중요도에 맞춰 조절 가능) ---
    w = np.array([1.7, 1.0, 1.0], dtype=np.float32)  # 예: pkt 덜 중요하면 [0.5,1.0,1.0] 등

    best_score = float('inf')   # 현재까지 최적 검증 점수, 초기값은 양의 무한대(inf)
    best_ep = -1 #최적 점수를 기록한 에폭 번호
    best_valid_huber = float('inf') #최적 검증 소실

    for epoch in range(1, args.epochs + 1): #어규먼트에서 받은 에폭 수 + 1 만큼 반복
        train_huber = run_epoch(dataloader_train, True)          # 학습 손실(Huber, mean)
        valid_huber = run_epoch(dataloader_valid, False)         # 검증 손실(Huber, mean)
        valid_mae_vector = epoch_mae_vec(dataloader_valid)       # [pkt, tcp, udp] MAE
        valid_score   = float((valid_mae_vector * w).sum())      # 저장 판단용 점수

        improved = valid_score < best_score #현재 에폭의 점수가 최적 점수보다 낮은지 확인
        if improved: 
            best_score = valid_score
            best_ep = epoch
            best_valid_huber = valid_huber
            state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_huber": valid_huber,
                "val_mae_vec": valid_mae_vector,
                "val_score": best_score,
                "weights": w,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "lookback": lookback,
                "horizon": horizon,
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale,
            }
            torch.save(state, args.ckpt) #state를 어규먼트에서 지정된 경로에 저장
            #매 에폭마다 출력
            print(f"[BEST] epoch={epoch} val_score(MAE·w)={valid_score:.4f} vec={np.round(valid_mae_vector, 4)}")

        if epoch % 5 == 0 or epoch == 1: #에폭 5번에 한번씩
            print(f"epoch: {epoch:02d} train(huber) {train_huber:.4f} | valid(huber) {valid_huber:.4f} | valid(MAE·sum) {valid_mae_vector.sum():.4f} | best@{best_ep} {best_score:.4f}")
#----------------------------------------------
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
