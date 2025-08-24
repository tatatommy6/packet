#패킷 캡처 데이터 전처리
import pandas as pd

csv = "data/house_cleaned.csv"

#wireshark로 패킷 캡처한 csv 파일을 불러오기
df = pd.read_csv(csv)

df['Time'] = pd.to_numeric(df['Time'], errors = 'coerce')

#10개를 한 묶음으로 만듦
df["Interval"] = (df["Time"] // 10).astype(int)

proto_counts = df.groupby(["Interval", "Protocol"]).size().unstack(fill_value=0)
#proto_counts 안에는 DNS,HTTP,ICMP,MDNS...등이 있음 (unstack() 때문에)

agg = df.groupby("Interval").agg(
    pkt_count=("Length", "count"),       # 패킷 개수
    len_mean=("Length", "mean"),         # 평균 길이
    len_std=("Length", "std"),           # 길이 표준편차
    len_max=("Length", "max"),           # 최대 길이
)
#agg 안에는 Interval, pkt_count, len_mean, len_std, len_max가 있음

# 두 데이터프레임을 합치는데 agg가 오른쪽에 proto_counts를 붙임
result =  agg.join(proto_counts)

# 비율 계산 (예: TCP/전체, UDP/전체)
for proto in ["TCP", "UDP"]:
    if proto in result.columns:
        result[f"{proto.lower()}_ratio"] = result[proto] / result["pkt_count"]

# 결측치를 0으로 채우기
result = result.fillna(0)

# 결과 확인 (디버깅용)
# print(result.head())

# CSV로 저장
result.to_csv("data/packetsby10s_9h.csv")


# --- 컬럼 설명 --- (by ChatGPT5)
# Interval   : 10초 단위 구간 번호 (0=0~10초, 1=10~20초 ...)
# pkt_count  : 해당 구간(10초) 동안 캡처된 패킷 개수
# len_mean   : 패킷 크기(Length)의 평균 (Byte 단위)
# len_std    : 패킷 크기의 표준편차 (패킷 크기 변동성)
# len_max    : 해당 구간에서 가장 큰 패킷 크기 (보통 MTU=1514 근처)
#
# DNS        : DNS 패킷 개수 (도메인 → IP 변환 요청)
# HTTP       : HTTP 패킷 개수 (암호화 안 된 웹 요청, 거의 드묾)
# ICMP       : ICMP 패킷 개수 (ping 프로토콜)
# MDNS       : Multicast DNS 패킷 (로컬 네트워크 장치 검색)
# QUIC       : QUIC 패킷 개수 (UDP 기반, HTTP/3, 유튜브/넷플릭스 등)
# SSLv2      : SSLv2 패킷 개수 (거의 사용되지 않음, 분석 큰 의미 없음)
# TCP        : TCP 패킷 개수 (웹, 다운로드, 전송 대부분)
# TLSv1.2    : TLS 1.2 패킷 개수 (구버전 HTTPS 트래픽)
# TLSv1.3    : TLS 1.3 패킷 개수 (최신 HTTPS 트래픽, 많이 사용됨)
# UDP        : UDP 패킷 개수 (게임, 스트리밍, 음성통화 등 실시간 서비스)
#
# tcp_ratio  : TCP 패킷 비율 (TCP / 전체 패킷 수)
# udp_ratio  : UDP 패킷 비율 (UDP / 전체 패킷 수)

#많은 컬럼이 생긴 이유: 14번 줄 때문인데 Protocol 컬럼에 여러 값이 들어있었고, unstack()이 그걸 전부 펼쳤기 때문.