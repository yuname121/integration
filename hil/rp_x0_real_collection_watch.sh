#!/usr/bin/env bash
# SafeNest RP-X0 real sensor collection observer.
# Classification: RP_X0_REAL_COLLECTION_OBSERVER
#
# Read-only. Does not send TCP/UDP sensor traffic, does not write evidence,
# does not restart the backend, and does not open the mmWave live B gate.

set -u

EXPECTED_SHA="1ffbc7d39792e68edc552fbe08359732b0dcbefd"
API_BASE="http://127.0.0.1:8000"
INTERVAL="2"
ONCE="0"
NO_CLEAR="0"
VERBOSE="0"
TAIL_N="50"
STALE_AFTER_S="10"
PHASE_STALE_MS="100"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

usage() {
  cat <<'EOF'
SafeNest RP-X0 Real Collection Watch
Classification: RP_X0_REAL_COLLECTION_OBSERVER

Usage:
  bash hil/rp_x0_real_collection_watch.sh
  bash hil/rp_x0_real_collection_watch.sh --interval 5
  bash hil/rp_x0_real_collection_watch.sh --once
  bash hil/rp_x0_real_collection_watch.sh --verbose
  bash hil/rp_x0_real_collection_watch.sh --no-clear
  bash hil/rp_x0_real_collection_watch.sh --help

Read-only observer for real ESP reception and persistence.
Does not send packets, write evidence, restart the backend, or open mmWave B.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      if [[ $# -lt 2 ]]; then
        echo "error: --interval requires a positive number" >&2
        exit 2
      fi
      INTERVAL="$2"
      shift 2
      ;;
    --once)
      ONCE="1"
      shift
      ;;
    --verbose|-v)
      VERBOSE="1"
      shift
      ;;
    --no-clear)
      NO_CLEAR="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${INTERVAL}" in
  ''|*[!0-9.]*)
    echo "error: --interval must be a positive number" >&2
    exit 2
    ;;
esac
awk -v n="${INTERVAL}" 'BEGIN { exit !(n + 0 > 0) }' || {
  echo "error: --interval must be a positive number" >&2
  exit 2
}

PREV_CO2_SIG=""
PREV_MMWAVE_SIG=""
PREV_THERMAL_SIG=""
THERMAL_INSPECT_SIG=""
THERMAL_INSPECT_TEXT="THERMAL_PERSISTENCE = NO_FILE_YET"

on_stop() {
  printf '\nwatch stopped\n'
  exit 0
}
# Reset ignored INT/TERM so Ctrl+C works in a real terminal.
trap - INT TERM
trap on_stop INT TERM

iso_now() {
  date '+%Y-%m-%dT%H:%M:%S%z' 2>/dev/null || date
}

detect_backend() {
  BACKEND_PID=""
  BACKEND_CMD="NONE"
  B_MODE="UNKNOWN"
  local line
  line="$(pgrep -af "backend/run_backend.py" 2>/dev/null | awk '!/pgrep/ && /run_backend\.py/ { print; exit }')"
  if [[ -z "${line}" ]]; then
    return 0
  fi
  BACKEND_PID="$(awk '{ print $1 }' <<<"${line}")"
  BACKEND_CMD="${line}"
  if [[ -n "${BACKEND_PID}" && -r "/proc/${BACKEND_PID}/environ" ]]; then
    local flag
    flag="$(tr '\0' '\n' <"/proc/${BACKEND_PID}/environ" 2>/dev/null | awk -F= '/^SAFENEST_RP_X0_B_RUNTIME=/ { print $2; exit }')"
    case "$(printf '%s' "${flag}" | tr '[:upper:]' '[:lower:]')" in
      1|true|yes|on) B_MODE="ON" ;;
      ''|0|false|no|off) B_MODE="OFF" ;;
      *) B_MODE="UNKNOWN" ;;
    esac
    if [[ -z "${flag}" ]]; then
      B_MODE="OFF"
    fi
  fi
}

detect_network() {
  TCP_LISTEN="NOT_LISTEN"
  TCP_CLASS="TCP_LISTENER_NOT_READY"
  TCP_ESTAB="none"
  UDP_BOUND="NOT_BOUND"
  UDP_CLASS="UDP_RECEIVER_NOT_BOUND"
  if ! command -v ss >/dev/null 2>&1; then
    TCP_LISTEN="ss_unavailable"
    UDP_BOUND="ss_unavailable"
    return 0
  fi
  if ss -ltn 2>/dev/null | awk '$4 ~ /:9000$/ { found=1 } END { exit !found }'; then
    TCP_LISTEN="LISTEN"
    TCP_CLASS="TCP_LISTENER_READY"
  fi
  local estab
  estab="$(ss -Htn state established 2>/dev/null | awk '
    $4 ~ /:9000$/ { print $5; next }
    $5 ~ /:9000$/ { print $4 }
  ')"
  if [[ -n "${estab}" ]]; then
    TCP_ESTAB="$(printf '%s' "${estab}" | tr '\n' '; ')"
    TCP_CLASS="TCP_ESP_CONNECTED"
  else
    TCP_ESTAB="none"
  fi
  if ss -uln 2>/dev/null | awk '$4 ~ /:5005$/ { found=1 } END { exit !found }'; then
    UDP_BOUND="BOUND"
    UDP_CLASS="UDP_RECEIVER_BOUND"
  fi
}

pi_health() {
  LOAD="$(uptime 2>/dev/null | sed -n 's/.*load averages*: //p')"
  RAM="unavailable"
  if [[ -r /proc/meminfo ]]; then
    RAM="$(awk '/MemAvailable:/ { printf "%.0f Mi available", $2/1024 }' /proc/meminfo)"
  elif command -v free >/dev/null 2>&1; then
    RAM="$(free -h 2>/dev/null | awk '/Mem:/ { print $7 " available" }')"
  fi
  RSS="unavailable"
  if [[ -n "${BACKEND_PID}" ]]; then
    RSS="$(ps -p "${BACKEND_PID}" -o rss= 2>/dev/null | awk '{ if ($1!="") printf "%s kB", $1 }')"
    [[ -n "${RSS}" ]] || RSS="unavailable"
  fi
  DISK="$(df -h "${REPO_ROOT}" 2>/dev/null | awk 'NR==2 { print $4 " free (" $5 " used)" }')"
  TEMP="unavailable"
  if command -v vcgencmd >/dev/null 2>&1; then
    TEMP="$(vcgencmd measure_temp 2>/dev/null | sed 's/temp=//')"
  elif [[ -r /sys/class/thermal/thermal_zone0/temp ]]; then
    TEMP="$(awk '{ printf "%.1fC", $1/1000 }' /sys/class/thermal/thermal_zone0/temp)"
  fi
}

file_sig() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'missing'
    return 0
  fi
  local size mtime
  size="$(wc -c <"${path}" 2>/dev/null | tr -d ' ')"
  mtime="$(stat -c '%Y' "${path}" 2>/dev/null || stat -f '%m' "${path}" 2>/dev/null || echo 0)"
  printf '%s:%s' "${size}" "${mtime}"
}

change_label() {
  local prev="$1"
  local now="$2"
  local mtime_s="$3"
  if [[ "${now}" == "missing" ]]; then
    printf 'NONE'
    return 0
  fi
  if [[ -z "${prev}" ]]; then
    printf 'NEW'
    return 0
  fi
  if [[ "${prev}" != "${now}" ]]; then
    printf 'NEW'
    return 0
  fi
  local now_epoch
  now_epoch="$(date +%s 2>/dev/null || echo 0)"
  if [[ -n "${mtime_s}" && "${mtime_s}" =~ ^[0-9]+$ && "${now_epoch}" =~ ^[0-9]+$ ]]; then
    local age=$((now_epoch - mtime_s))
    if (( age > STALE_AFTER_S )); then
      printf 'NO_RECENT_UPDATE %ss' "${age}"
      return 0
    fi
  fi
  printf 'UNCHANGED'
}

latest_match() {
  local dir="$1"
  local pattern="$2"
  local latest=""
  local latest_m=0
  local f m
  shopt -s nullglob
  for f in "${dir}"/${pattern}; do
    [[ -f "${f}" ]] || continue
    [[ "$(basename "${f}")" == ".gitkeep" ]] && continue
    m="$(stat -c '%Y' "${f}" 2>/dev/null || stat -f '%m' "${f}" 2>/dev/null || echo 0)"
    m="${m%.*}"
    m="${m:-0}"
    if [[ "${m}" -ge "${latest_m}" ]]; then
      latest="${f}"
      latest_m="${m}"
    fi
  done
  shopt -u nullglob
  printf '%s' "${latest}"
}

inspect_thermal() {
  local path="$1"
  if [[ -z "${path}" || ! -f "${path}" ]]; then
    THERMAL_INSPECT_TEXT="THERMAL_PERSISTENCE = NO_FILE_YET"
    return 0
  fi
  if [[ -z "${PYTHON_BIN}" ]]; then
    THERMAL_INSPECT_TEXT="npz inspect: python unavailable"
    return 0
  fi
  THERMAL_INSPECT_TEXT="$(
    "${PYTHON_BIN}" - "${path}" <<'PY' 2>/dev/null || echo "npz inspect failed"
import sys
path = sys.argv[1]
try:
    import numpy as np
except Exception as error:
    print(f"numpy unavailable: {error}")
    raise SystemExit(0)
try:
    with np.load(path, allow_pickle=False) as saved:
        frames = saved["frames"] if "frames" in saved.files else None
        seq = saved["frame_sequences"] if "frame_sequences" in saved.files else None
        ts = saved["timestamps"] if "timestamps" in saved.files else None
        recv = saved["receive_monotonic"] if "receive_monotonic" in saved.files else None
        mn = saved["minimum_raw"] if "minimum_raw" in saved.files else None
        mx = saved["maximum_raw"] if "maximum_raw" in saved.files else None
        shape = list(frames.shape) if frames is not None else None
        dtype = str(frames.dtype) if frames is not None else None
        geom = "OK" if shape is not None and len(shape) == 3 and shape[1:] == [62, 80] and dtype == "uint16" else "UNEXPECTED"
        print(
            "shape={shape} dtype={dtype} geom={geom} n={n}".format(
                shape=shape,
                dtype=dtype,
                geom=geom,
                n=None if frames is None else int(frames.shape[0]),
            )
        )
except Exception as error:
    print(f"npz inspect failed: {type(error).__name__}: {error}")
PY
  )"
}

render_sensor_python() {
  local health_json="$1"
  local status_json="$2"
  local co2_path="$3"
  local mmwave_path="$4"
  [[ -n "${PYTHON_BIN}" ]] || {
    echo "python unavailable; skipping JSON/file extraction"
    return 0
  }
  local wrapper
  wrapper="$(
    printf '{"health":'
    if [[ -n "${health_json}" ]]; then printf '%s' "${health_json}"; else printf 'null'; fi
    printf ',"status":'
    if [[ -n "${status_json}" ]]; then printf '%s' "${status_json}"; else printf 'null'; fi
    printf '}'
  )"
  WATCH_API_JSON="${wrapper}" WATCH_VERBOSE="${VERBOSE}" "${PYTHON_BIN}" - "${co2_path}" "${mmwave_path}" "${TAIL_N}" "${PHASE_STALE_MS}" <<'PY' || echo "python extract failed"
import json
import math
import os
import sys
from pathlib import Path

co2_path = sys.argv[1]
mmwave_path = sys.argv[2]
tail_n = int(sys.argv[3])
phase_stale_ms = int(sys.argv[4])
payload = json.loads(os.environ.get("WATCH_API_JSON") or "null")
if not isinstance(payload, dict):
    payload = {}
health = payload.get("health")
status = payload.get("status")
MISSING = object()


def mapping(value):
    return value if isinstance(value, dict) else {}


def short(value):
    if value is MISSING:
        return "absent"
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def gate(value):
    if value is MISSING:
        return "NO"
    if value is None:
        return "NULL"
    return "YES"


def walk(row, *names, nested_only=False):
    nested = mapping(row.get("mmwave"))
    for name in names:
        if name in nested:
            return nested.get(name)
        if not nested_only and name in row:
            return row.get(name)
    return MISSING


def tail_lines(path, limit):
    target = Path(path)
    if path in {"", "NONE"} or not target.is_file():
        return []
    try:
        with target.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            data = b""
            while size > 0 and data.count(b"\n") <= limit:
                step = min(8192, size)
                size -= step
                handle.seek(size)
                data = handle.read(step) + data
        return data.decode("utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def load_tail(path, limit):
    rows = []
    for line in tail_lines(path, limit):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def cadence(values):
    numeric = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    deltas = [b - a for a, b in zip(numeric, numeric[1:]) if b > a]
    med = median(deltas)
    hz = None if med in (None, 0) else round(1.0 / med, 3)
    return len(numeric), med, hz


verbose = os.environ.get("WATCH_VERBOSE") == "1"
alerts = []


def note(flag):
    if flag and flag not in alerts:
        alerts.append(flag)


health_ok = "UNAVAILABLE"
ready = "UNAVAILABLE"
last_error_s = "UNAVAILABLE"
receiver = {}
logger = {}
udp = {}
if isinstance(health, dict):
    health_ok = health.get("ok")
    ready = health.get("ready")
    last_error = health.get("last_error")
    receiver = mapping(health.get("receiver"))
    logger = mapping(receiver.get("sensor_logging"))
    udp = mapping(receiver.get("thermal_udp"))
    if isinstance(last_error, dict):
        details = mapping(last_error.get("details"))
        last_error_s = "{src}: {detail}".format(
            src=details.get("source"),
            detail=details.get("detail"),
        )
        if "deadline" in str(details.get("detail") or "").lower():
            note("RECEIVE_DEADLINE")
    else:
        last_error_s = "none"
    if isinstance(receiver.get("protocol_errors"), int) and receiver.get("protocol_errors") > 0:
        note("PROTOCOL_ERRORS")
    if isinstance(receiver.get("disconnects"), int) and receiver.get("disconnects") > 0:
        note("TCP_DISCONNECTS")

dropped = mapping(logger.get("dropped"))
drop_total = sum(int(v or 0) for v in dropped.values() if isinstance(v, (int, float))) if dropped else 0
qsize = logger.get("queue_size")
qcap = logger.get("queue_capacity")
errors = logger.get("errors")
if drop_total > 0:
    note("LOGGER_DROPS")
if isinstance(errors, int) and errors > 0:
    note("LOGGER_ERRORS")
if isinstance(qsize, int) and isinstance(qcap, int) and qcap > 0 and qsize >= int(0.8 * qcap):
    note("QUEUE_NEAR_CAPACITY")

if co2_path in {"", "NONE"}:
    last_co2 = None
    co2_line = "NO_PERSISTED_CO2_YET"
else:
    last_co2 = (load_tail(co2_path, 1) or [None])[-1]
    if last_co2 is None:
        co2_line = "EMPTY"
    else:
        boot = last_co2["boot_id"] if "boot_id" in last_co2 else MISSING
        event_id = last_co2["co2_measurement_event_id"] if "co2_measurement_event_id" in last_co2 else MISSING
        event_ms = last_co2["co2_measurement_monotonic_ms"] if "co2_measurement_monotonic_ms" in last_co2 else MISSING
        prov = "PRESENT" if gate(boot) == "YES" and gate(event_id) == "YES" and gate(event_ms) == "YES" else "INCOMPLETE"
        co2_line = "ppm={ppm}  prov={prov}  boot={boot} event={eid} ms={ems}".format(
            ppm=short(last_co2["co2_ppm"] if "co2_ppm" in last_co2 else MISSING),
            prov=prov,
            boot=gate(boot),
            eid=gate(event_id),
            ems=gate(event_ms),
        )

co2_ai = mapping(mapping(status).get("co2")).get("ai") if isinstance(status, dict) else None
if not isinstance(co2_ai, dict):
    co2_b = "API_UNAVAILABLE"
else:
    md = mapping(co2_ai.get("metadata"))
    occupied = md.get("occupied_probability")
    err = co2_ai.get("error") or co2_ai.get("state")
    co2_b = "C-B6 {cand}  {err}".format(
        cand=md.get("candidate_id") or co2_ai.get("model_id") or "?",
        err=err,
    )
    if isinstance(occupied, (int, float)):
        decision = co2_ai.get("state") if co2_ai.get("state") in {"OCCUPIED", "VACANT"} else (
            "OCCUPIED" if occupied >= 0.43 else "VACANT"
        )
        co2_b += f"  P={occupied:.3g} thr=0.43 {decision}"

if mmwave_path in {"", "NONE"}:
    rows = []
    mm_line = "NO_PERSISTED_MMWAVE_YET"
else:
    rows = load_tail(mmwave_path, tail_n)

    def counted(name):
        n = 0
        for row in rows:
            value = walk(row, name)
            if value is not MISSING and value is not None:
                n += 1
        return n

    recent = len(rows)
    phase_n = counted("breath_phase")
    contract = "OBSERVED" if phase_n else "NOT_OBSERVED"
    recv_n, recv_med, recv_hz = cadence([row.get("receive_monotonic") for row in rows])
    src_vals = [walk(row, "ts_monotonic_ms") for row in rows]
    src_n, src_med, src_hz = cadence([v for v in src_vals if v is not MISSING])
    ages = [int(walk(row, "phase_age_ms")) for row in rows if isinstance(walk(row, "phase_age_ms"), (int, float))]
    if ages and median(ages) is not None and median(ages) >= phase_stale_ms and recv_med is not None and recv_med < 0.5:
        note("PHASE_AGE_STALE")
    latest = rows[-1] if rows else None
    mm_line = "phase={p}/{r} {c}  PiHz={hz}  ESPHz={esp}  seq={seq}".format(
        p=phase_n,
        r=recent,
        c=contract,
        hz=recv_hz,
        esp=(None if not src_n or src_med in (None, 0) else round(1000.0 / src_med, 3)),
        seq=short(latest["sequence"] if latest and "sequence" in latest else MISSING),
    )
    if ages:
        mm_line += f"  age={ages[-1]}(med={median(ages)})"

mm_ai = mapping(mapping(status).get("mmwave")).get("ai") if isinstance(status, dict) else None
if not isinstance(mm_ai, dict):
    mm_gate = "UNAVAILABLE"
else:
    md = mapping(mm_ai.get("metadata"))
    mm_gate = "{gate}  {reason}".format(
        gate=md.get("live_gate") or mm_ai.get("error") or "UNKNOWN",
        reason=md.get("live_gate_reason") or "",
    )

th_ai = mapping(mapping(status).get("thermal")).get("ai") if isinstance(status, dict) else None
if not isinstance(th_ai, dict):
    th_b = "UNAVAILABLE"
else:
    md = mapping(th_ai.get("metadata"))
    th_b = "{err}  artifact={art}  hist_minmax={hist}".format(
        err=th_ai.get("error") or th_ai.get("state"),
        art=md.get("artifact_present"),
        hist=md.get("historical_minmax_used"),
    )

pir_ai = mapping(mapping(status).get("pir")).get("ai") if isinstance(status, dict) else None
pir_state = mapping(mapping(status).get("pir")).get("state") if isinstance(status, dict) else None
values = mapping(mapping(pir_state).get("values"))
latest_mm = rows[-1] if rows else {}
if "motion" in values:
    motion = values.get("motion")
elif "pir_motion" in latest_mm:
    motion = latest_mm.get("pir_motion")
else:
    motion = MISSING
pir_line = f"motion={short(motion)}  ai={mapping(pir_ai).get('state')}"

print(f"[API] health={health_ok} ready={ready}  last_error={last_error_s}")
print(
    "[LOGGER] q={q}/{c}  drop={d}  err={e}  mmw={mm} co2={co2} th={th}".format(
        q="?" if qsize is None else qsize,
        c="?" if qcap is None else qcap,
        d=drop_total,
        e="?" if errors is None else errors,
        mm=mapping(logger.get("written")).get("mmwave"),
        co2=mapping(logger.get("written")).get("co2"),
        th=mapping(logger.get("written")).get("thermal"),
    )
)
print(f"[CO2] {co2_line}")
print(f"      {co2_b}")
print(f"[MMWAVE] {mm_line}")
print(f"         live B gate {mm_gate}")
print(f"[THERMAL B] {th_b}")
print(f"[PIR] {pir_line}")
if udp:
    print(
        "[UDP] datagrams={d} frames={c} incomplete={i} crc={k} timeout={t}".format(
            d=udp.get("received_datagrams"),
            c=udp.get("completed_frames"),
            i=udp.get("incomplete_frames"),
            k=udp.get("checksum_failures"),
            t=udp.get("reconstruction_timeouts"),
        )
    )
if alerts:
    print("[ALERTS] " + "  ".join(alerts))

if verbose and last_co2:
    print(
        "[CO2 DETAIL] device={d} seq={s} uptime={u}".format(
            d=last_co2.get("device_id"),
            s=last_co2.get("sequence"),
            u=last_co2.get("source_uptime_ms"),
        )
    )
if verbose and rows:
    latest = rows[-1]
    print(
        "[MMWAVE DETAIL] device={d} resp={r} heart={h} phase={p} fw={fw}".format(
            d=latest.get("device_id"),
            r=short(latest["respiration_rate_bpm"] if "respiration_rate_bpm" in latest else MISSING),
            h=short(latest["heart_rate_bpm"] if "heart_rate_bpm" in latest else MISSING),
            p=short(walk(latest, "breath_phase")),
            fw=short(walk(latest, "firmware_version")),
        )
    )
PY
}

render() {
  detect_backend
  detect_network
  pi_health

  local health_json="" status_json=""
  health_json="$(curl -fsS --max-time 1 "${API_BASE}/health" 2>/dev/null || true)"
  status_json="$(curl -fsS --max-time 1 "${API_BASE}/api/status" 2>/dev/null || true)"

  local co2_file mmwave_file thermal_file
  co2_file="$(latest_match "${DATA_ROOT}/co2" "*.jsonl")"
  mmwave_file="$(latest_match "${DATA_ROOT}/mmwave" "*.jsonl")"
  thermal_file="$(latest_match "${DATA_ROOT}/thermal" "*.npz")"
  [[ -n "${co2_file}" ]] || co2_file="NONE"
  [[ -n "${mmwave_file}" ]] || mmwave_file="NONE"
  [[ -n "${thermal_file}" ]] || thermal_file="NONE"

  local co2_sig mmwave_sig thermal_sig co2_mtime mmwave_mtime thermal_mtime
  co2_sig="$(file_sig "${co2_file}")"
  mmwave_sig="$(file_sig "${mmwave_file}")"
  thermal_sig="$(file_sig "${thermal_file}")"
  co2_mtime="$(stat -c '%Y' "${co2_file}" 2>/dev/null || stat -f '%m' "${co2_file}" 2>/dev/null || echo "")"
  mmwave_mtime="$(stat -c '%Y' "${mmwave_file}" 2>/dev/null || stat -f '%m' "${mmwave_file}" 2>/dev/null || echo "")"
  thermal_mtime="$(stat -c '%Y' "${thermal_file}" 2>/dev/null || stat -f '%m' "${thermal_file}" 2>/dev/null || echo "")"

  if [[ "${thermal_file}" != "NONE" && "${thermal_sig}" != "${THERMAL_INSPECT_SIG}" ]]; then
    inspect_thermal "${thermal_file}"
    THERMAL_INSPECT_SIG="${thermal_sig}"
  fi
  if [[ "${thermal_file}" == "NONE" ]]; then
    THERMAL_INSPECT_TEXT="THERMAL_PERSISTENCE = NO_FILE_YET"
    THERMAL_INSPECT_SIG=""
  fi

  local sha warn=""
  sha="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo UNAVAILABLE)"
  if [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)" != "${EXPECTED_SHA}" ]]; then
    warn="  SHA_DIFFERS"
  fi
  echo "RP-X0 watch  $(iso_now)  sha=${sha}  B=${B_MODE}${warn}"
  echo "[RUNTIME] PID=${BACKEND_PID:-NONE}  ${TCP_CLASS}  UDP=${UDP_CLASS}"
  echo "[NETWORK] TCP9000 ${TCP_LISTEN}  ESP ${TCP_ESTAB:-none}  UDP5005 ${UDP_BOUND}"
  if [[ "${co2_file}" == "NONE" ]]; then
    echo "[CO2 FILE] NONE"
  else
    echo "[CO2 FILE] $(basename "${co2_file}")  n=$(wc -l <"${co2_file}" | tr -d ' ')  $(change_label "${PREV_CO2_SIG}" "${co2_sig}" "${co2_mtime}")"
  fi
  if [[ "${mmwave_file}" == "NONE" ]]; then
    echo "[MMWAVE FILE] NONE"
  else
    echo "[MMWAVE FILE] $(basename "${mmwave_file}")  n=$(wc -l <"${mmwave_file}" | tr -d ' ')  $(change_label "${PREV_MMWAVE_SIG}" "${mmwave_sig}" "${mmwave_mtime}")"
  fi
  if [[ "${thermal_file}" == "NONE" ]]; then
    echo "[THERMAL] NO_FILE_YET"
  else
    echo "[THERMAL] $(basename "${thermal_file}")  $(change_label "${PREV_THERMAL_SIG}" "${thermal_sig}" "${thermal_mtime}")  ${THERMAL_INSPECT_TEXT}"
  fi
  echo "[PI] load ${LOAD:-?}  RAM ${RAM}  RSS ${RSS}  temp ${TEMP}  disk ${DISK:-?}"

  render_sensor_python "${health_json}" "${status_json}" "${co2_file}" "${mmwave_file}"

  PREV_CO2_SIG="${co2_sig}"
  PREV_MMWAVE_SIG="${mmwave_sig}"
  PREV_THERMAL_SIG="${thermal_sig}"
}

if [[ "${ONCE}" == "1" ]]; then
  render
  exit 0
fi

while true; do
  if [[ "${NO_CLEAR}" != "1" ]] && [[ -t 1 ]]; then
    printf '\033[H\033[2J'
  fi
  render
  sleep "${INTERVAL}"
done
