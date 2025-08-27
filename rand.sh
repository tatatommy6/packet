# 데이터 수집할 때 썻던 네트워크 자동 부하를 주는 bash파일
# - 2시간: iperf 랜덤(라이트) 부하 + iperf 전용 캡처(헤더만, 롤링)
# - 이후 6시간: 자연 트래픽 캡처(헤더만, 롤링, 시간 상한)
# * dumpcap은 Wireshark 설치 시 포함. 권한 요구 시 macOS 암호 입력.
# 사용: ./safe_net_capture.sh <RPI_IP> [IFACE]
# 예:  ./safe_net_capture.sh 172.30.1.79 en0

set -euo pipefail

RPI_IP="${1:-}"
IFACE="${2:-en0}"                     # 기본 Wi‑Fi 인터페이스 가정
[ -z "$RPI_IP" ] && { echo "사용법: $0 <RPI_IP> [IFACE]"; exit 1; }

# ==== 설정값(필요시 조정) ===================================================
IPERF_DURATION=$((2*60*60))           # 2시간(초)
NATURAL_DURATION=$((6*60*60))         # 6시간(초) - 자연 캡처 상한
PORT=5201

# iperf 라이트(UDP 권장)
MIN_BW=3;   MAX_BW=30                 # Mbps
MIN_BURST=10; MAX_BURST=40            # 송출 10~40초
MIN_REST=20;  MAX_REST=120            # 휴식 20~120초
MIN_PAR=1;   MAX_PAR=2                # 동시 스트림 1~2
PROTO="udp"                           # "udp" 또는 "tcp"

# 캡처(헤더만 + 롤링 제한)
SNAPLEN=96                            # 헤더만 저장(개인정보/용량 부담↓)
IPERF_FILESIZE_MB=150                 # iperf 캡처 파일 당 MB
IPERF_FILES_MAX=40                    # 파일 개수 한도(= 최대 약 6GB)
NAT_ROTATE_SEC=300                    # 자연 캡처 5분 단위 파일 회전
NAT_FILES_MAX=72                      # 72개 * 5분 = 6시간치 유지
# ============================================================================

timestamp() { date '+%Y-%m-%d_%H-%M-%S'; }
rand_int() { local a=$1 b=$2; echo $(( RANDOM % (b - a + 1) + a )); }

# 경로 준비
BASE="$HOME/captures/$(timestamp)"
mkdir -p "$BASE"
echo "저장 경로: $BASE"

# 내 IP 확인
MY_IP=$(ipconfig getifaddr "$IFACE" || true)
if [ -z "${MY_IP:-}" ]; then
  echo "⚠️  인터페이스 $IFACE 에서 IP를 구하지 못했어요. 계속 진행은 가능하지만 자연 캡처 필터에 내 IP가 반영되지 않을 수 있어요."
fi

# 자식 프로세스 정리
PIDS=()
cleanup() {
  echo; echo "정리 중..."
  for p in "${PIDS[@]:-}"; do
    kill "$p" 2>/dev/null || true
    wait "$p" 2>/dev/null || true
  done
  echo "끝."
}
trap cleanup EXIT INT TERM

# 1) iperf 전용 캡처 시작 (헤더만 + 필터 + 롤링)
echo "[1/3] iperf 캡처 시작(dumpcap)…"
sudo dumpcap -i "$IFACE" \
  -f "host $RPI_IP and (tcp port $PORT or udp port $PORT) and not broadcast and not multicast" \
  -s "$SNAPLEN" \
  -b "filesize:$IPERF_FILESIZE_MB" -b "files:$IPERF_FILES_MAX" \
  -w "$BASE/iperf-roll.pcapng" \
  >/dev/null 2>&1 &
PIDS+=($!)
sleep 1

# 2) 2시간 iperf 랜덤(라이트) 부하
echo "[2/3] iperf 랜덤(라이트) 부하 시작(총 ${IPERF_DURATION}s)…"
END=$(( $(date +%s) + IPERF_DURATION ))
while [ "$(date +%s)" -lt "$END" ]; do
  BW=$(rand_int $MIN_BW $MAX_BW)           # Mbps
  BURST=$(rand_int $MIN_BURST $MAX_BURST)  # sec
  REST=$(rand_int $MIN_REST $MAX_REST)     # sec
  PAR=$(rand_int $MIN_PAR $MAX_PAR)

  echo "▶ $(date '+%H:%M:%S') | ${BW}Mbps, ${BURST}s, ${PAR}streams ($PROTO)"
  if [ "$PROTO" = "udp" ]; then
    iperf3 -c "$RPI_IP" -u -b "${BW}M" -p "$PORT" -t "$BURST" -P "$PAR" >/dev/null 2>&1
  else
    iperf3 -c "$RPI_IP" --bitrate "${BW}M" -p "$PORT" -t "$BURST" -P "$PAR" >/dev/null 2>&1 \
    || iperf3 -c "$RPI_IP" -p "$PORT" -t "$BURST" -P "$PAR" >/dev/null 2>&1
  fi
  echo "⏸ 휴식 ${REST}s"; sleep "$REST"
done
echo "…iperf 부하 종료."

# iperf 전용 캡처 중단
echo "iperf 캡처 중단…"
sudo pkill -f "dumpcap -i $IFACE -f host $RPI_IP" >/dev/null 2>&1 || true
sleep 1

# 3) 자연 트래픽 캡처 시작(헤더만 + 5분 회전 + 6시간 상한)
if [ -n "${MY_IP:-}" ]; then
  NAT_FILTER="host $MY_IP and not port 53 and not port 5353 and not multicast and not broadcast"
else
  # 내 IP를 못 얻었으면 인터페이스 전체(주의: 데이터 많아짐)
  NAT_FILTER="not port 53 and not port 5353 and not multicast and not broadcast"
fi

echo "[3/3] 자연 트래픽 캡처 시작(최대 ${NATURAL_DURATION}s)…"
sudo dumpcap -i "$IFACE" \
  -f "$NAT_FILTER" \
  -s "$SNAPLEN" \
  -b "duration:$NAT_ROTATE_SEC" -b "files:$NAT_FILES_MAX" \
  -a "duration:$NATURAL_DURATION" \
  -w "$BASE/natural-roll.pcapng" \
  >/dev/null 2>&1 &
PIDS+=($!)

echo
echo "✅ 실행 중입니다. 수면 방지를 원하면 다음처럼 실행하세요:"
echo "   caffeinate -dimsu $0 $RPI_IP $IFACE"
echo
echo "📂 캡처 파일은 여기 누적됩니다: $BASE"
echo "   - iperf 구간:  iperf-roll.pcapng (파일 회전, 최대 ~${IPERF_FILESIZE_MB}MB x ${IPERF_FILES_MAX}개)"
echo "   - 자연 구간:   natural-roll.pcapng (5분 회전, 최대 ${NAT_FILES_MAX}개 = ~6시간)"
echo
echo "⛔️ 중간에 종료하려면:  Ctrl+C"
wait