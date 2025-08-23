# 📡 Network QoS Prediction with LSTM

> **카페/가정 네트워크 환경에서 패킷 데이터를 수집하고, 10초 단위로 집계한 후  
> LSTM 모델을 이용해 QoS(패킷 수, TCP/UDP 비율, 지연 등)를 예측하는 프로젝트**

---

## 📑 Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Results](#results)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [License](#license)

---

## 🔎 Introduction
- 공용/가정망에서 발생하는 네트워크 품질(QoS) 변동을 시계열 관점에서 분석
- 패킷 단위 데이터를 10초 구간으로 집계 → LSTM 데이터셋으로 사용
- 목표: 다음 10초 구간의 트래픽 특성(패킷 수, TCP/UDP 비율 등) 또는 RTT(네트워크 왕복 시간) 예측

---

## 📂 Dataset
- **수집 도구**: Wireshark
- **원본 포맷**: `.pcapng`
- **가공 포맷**: `.csv`
- **집계 방식**: 10초 단위 Interval 집계
  - `pkt_count`: 패킷 개수
  - `len_mean`, `len_std`, `len_max`
  - `Protocol counts`: TCP, UDP, DNS, QUIC, TLSv1.3 ...
  - `tcp_ratio`, `udp_ratio`

---

## 🛠 Preprocessing
1. Wireshark CSV export (`No, Time, Source, Destination, Protocol, Length, Info`)
2. 10초 단위 Interval 집계 (pandas)
   
--현재 여기까지 완료--

3. Feature scaling (`StandardScaler`)
4. Dataset split (Train 70%, Valid 15%, Test 15%, 시간순 유지)

