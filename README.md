# 📡 DDoS Early Warning with LSTM

> **This is a lightweight program that predicts DDoS in advance
by aggregating packets into 10-second utils in café, home and campus networks,
predicting traffic characteristics of the next time window using an LSTM model**

> *(Originaly it was a QoS(network quality delivered to end devices) predict project, bui it later pivoted into DDoS early warning project based on learning and analysis results)*

---

## 🔎 Introduction
- Analyzed time-series changes in traffic composition(especially UDP) in public and indoor.
- collect packet data in **10-second windows** -> convert to a sequence **(N, T, F)** to be processed by LSTM 
- **goal:** predict `pkt_count`, `tcp_ratio`, `udp_ratio` in next 10 seconds and use them as triggers for early warning.
- **current status**  
  - model training complete
  - 테스트셋에서 **비율 예측 MAE -> 0.10~0.12** 확인
  - The **prediction MAE was measured at approximately 0.10–0.12** on the test set.
  - **Real-world testing** has not yet begun.
  
---

## 🎯 Purpose
- **네트워크 다운타임 -> 서비스 중단 등 직접적인 손실 발생**  
- 중소규모 환경(소상공인, 학원, 동호회 서버 등)도 **UDP/소형 패킷 급증** 같은 **공격 전조**가 발생할 수 있음  
- 상용 방어 제품은 “발생 시 완화”에 강점 → 본 프로젝트는 **온프레미스에서 가볍게 돌릴 수 있는 조기경보**에 초점

---

## 📈 Dataset creation process
- 라즈베리파이로 iperf3을 이용해 서버를 연 후, 클라이언트 쪽에서 서버에 부하를 2시간 동안 랜덤하게 주어 다양한 형태의 데이터 수집 (Wireshark 활용)
- 이후 6시간 동안 일부러 부하를 주지 않으면서(집안의 다양한 기기들의 트래픽만 받음) 평상시의 데이터도 수집 (Wireshark 활용)
- 총 26840초(7.9시간)동안 428326개의 패킷 수집 (Wireshark 활용)

---

## 🧱 How It Works
1. **데이터 수집 과정 & 집계**
   - wireshark 로 캡처한 패킷을 10초에 한묶음으로 만듦
   - 집계 지표: `pkt_count`, `len_mean`, `len_std`, `len_max`, `tcp_ratio`, `udp_ratio` + 주요 프로토콜 비율  

3. **전처리 & 시퀀스 변환**  
   - `.csv` → `.npz` 변환  
   - 슬라이딩 윈도우로 `(샘플 수 N, lookback T, feature 수 F)` 시퀀스 생성  
   - StandardScaler로 정규화  

4. **모델 학습 (PyTorch LSTM)**  
   - 입력: 과거 lookback 구간 (예: 24스텝 = 240초)  
   - 출력: 다음 구간의 QoS 지표  
   - 손실: HuberLoss, 최적화: Adam  

5. **성능 평가**  
   - Validation: MAE 기준 **tcp/udp ≈ 0.10 ~ 0.12**  
   - Baseline 대비 **80% 이상 향상**  

---

## 🚧 Current Status
- LSTM 모델 구현 및 학습 완료  
- 테스트셋 성능 검증 완료  
- 실시간 네트워크에서의 **실전 테스트는 진행 전**  
- 추후 계획: 실시간 경보 시스템(예: 알림, 차단 룰 연동) 적용
