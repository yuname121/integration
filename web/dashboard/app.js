"use strict";

const $ = (id) => document.getElementById(id);
const state = { publication: null, socket: null, pollTimer: null, reconnectTimer: null, auxiliaryAt: 0 };

const reasonLabels = {
  EMERGENCY_HUMAN_FALL: "Thermal AI가 높은 신뢰도로 낙상을 감지했습니다.",
  EMERGENCY_VERIFIED_APNEA: "검증된 무호흡 신호가 감지되었습니다.",
  APNEA_UNVERIFIED_NO_OVERRIDE: "무호흡 후보가 있으나 검증되지 않아 긴급 승격하지 않았습니다.",
  ABNORMAL_RESPIRATION_RPM: "호흡수가 정상 범위(12–20 rpm)를 벗어났습니다.",
  HIGH_CO2_WARNING: "CO₂ 농도가 주의 기준 이상입니다.",
  HIGH_CO2_DANGER: "CO₂ 농도가 위험 기준 이상입니다.",
  FAST_CO2_RISE: "CO₂ 농도가 빠르게 상승하고 있습니다.",
  LONG_NO_MOTION: "사람이 확인된 상태에서 15초 이상 움직임이 없습니다.",
  NO_MOTION_DETECTED: "사람이 확인되었지만 현재 움직임이 없습니다.",
  PRESENCE_NOT_CONFIRMED: "사람 존재가 확인되지 않아 무움직임 위험을 누적하지 않습니다.",
  PRESENCE_UNCONFIRMED: "mmWave presence 입력을 사용할 수 없습니다.",
  PRESENCE_FROM_THERMAL: "Thermal AI로 사람 존재를 교차 확인했습니다.",
  PRESENCE_FROM_MMWAVE: "mmWave로 사람 존재를 확인했습니다.",
  MMWAVE_THERMAL_MISMATCH: "mmWave와 Thermal의 사람 존재 판정이 일치하지 않습니다.",
  ALL_RISK_COMPONENTS_UNAVAILABLE: "사용 가능한 위험도 입력이 없습니다.",
  ALL_SENSORS_FAULT_OR_MISSING: "모든 센서가 결측 또는 고장 상태입니다."
};

const eventLabels = {
  SNAPSHOT_INITIALIZED: "관제 상태 초기화",
  RISK_LEVEL_CHANGED: "위험 단계 변경",
  SYSTEM_HEALTH_CHANGED: "시스템 건강도 변경",
  EMERGENCY_STARTED: "긴급 경보 시작",
  EMERGENCY_CLEARED: "긴급 경보 해제",
  SENSOR_STATUS_CHANGED: "센서 상태 변경",
  RUNTIME_ERROR: "Runtime 오류"
};

function number(value, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function percent(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

function setText(id, value) { $(id).textContent = value ?? "—"; }

function setStatus(id, status) {
  const value = status || "NO DATA";
  const element = $(id);
  element.textContent = value.replaceAll("_", " ");
  element.dataset.status = value;
}

function valueAt(sensor, key) { return sensor?.state?.values?.[key]; }
function aiAt(sensor) { return sensor?.ai || {}; }
function componentAt(sensor) { return sensor?.risk_component || {}; }

function riskComponentText(sensor) {
  const component = componentAt(sensor);
  if (!component.available) return "UNAVAILABLE";
  return `${number(component.score, 2)} · ${(component.source || "UNKNOWN").toUpperCase()}`;
}

function renderOverview(payload) {
  const risk = payload.risk || {};
  const level = risk.risk_level || "UNKNOWN";
  const score = typeof risk.risk_score === "number" ? risk.risk_score : null;
  document.body.dataset.level = level;
  setText("riskLevel", level === "UNKNOWN" ? "NO DATA" : level);
  setText("riskKicker", risk.is_emergency ? "즉시 현장 확인 필요" : level === "UNKNOWN" ? "데이터 수신 대기" : "실시간 융합 위험도");
  const summary = {
    NORMAL: "현재 사용 가능한 센서 기준으로 즉시 대응이 필요한 위험은 없습니다.",
    WARNING: "주의 신호가 감지되었습니다. 원인 센서와 현장 상태를 확인하세요.",
    DANGER: "위험 신호가 확인되었습니다. 즉시 현장 대응 절차를 시작하세요.",
    UNKNOWN: "유효한 센서 입력이 부족해 위험도를 계산할 수 없습니다."
  }[level];
  setText("riskSummary", summary);
  setText("riskScore", score === null ? "—" : Math.round(score));
  $("scoreGauge").style.setProperty("--score", score === null ? 0 : Math.max(0, Math.min(100, score)));
  $("scoreGauge").setAttribute("aria-label", score === null ? "위험 점수 없음" : `위험 점수 ${score.toFixed(1)}점`);
  setText("gatewayState", payload.system || "OFFLINE");
  setText("stateRevision", payload.revision ?? "—");
  setText("lastUpdate", formatTime(payload.timestamp));
  const health = payload.system_health || "FAILED";
  setText("healthBadge", health);
  $("healthBadge").dataset.health = health;
  renderReasons(risk.reasons || []);
}

function renderReasons(reasons) {
  const list = $("reasonList");
  list.replaceChildren();
  const unique = [...new Set(reasons)];
  setText("reasonCount", unique.length);
  if (!unique.length) {
    const item = document.createElement("li");
    item.className = "empty-state";
    item.textContent = "현재 수집된 판단 근거가 없습니다.";
    list.append(item);
    return;
  }
  unique.slice(0, 8).forEach((reason) => {
    const item = document.createElement("li");
    item.textContent = reasonLabels[reason] || reason.replaceAll("_", " ");
    if (reason.startsWith("EMERGENCY") || reason.includes("DANGER")) item.className = "critical";
    list.append(item);
  });
}

function renderSensors(payload) {
  const mmwave = payload.mmwave || {};
  setStatus("mmwaveStatus", mmwave.state?.status);
  setText("respirationValue", number(valueAt(mmwave, "respiration_rate_bpm"), 1));
  setText("heartValue", number(valueAt(mmwave, "heart_rate_bpm"), 1));
  const presenceAvailable = valueAt(mmwave, "presence_available") === true;
  const fusedPresence = payload.risk?.presence_detected;
  const fusedSource = payload.risk?.presence_source;
  setText("presenceValue", presenceAvailable
    ? (valueAt(mmwave, "presence") ? "감지 · mmWave" : "미감지 · mmWave")
    : typeof fusedPresence === "boolean" && fusedSource !== "UNCONFIRMED"
      ? `${fusedPresence ? "감지" : "미감지"} · ${fusedSource}`
      : "미제공");
  setText("mmwaveAi", aiAt(mmwave).available ? aiAt(mmwave).state : (aiAt(mmwave).error || "INPUT UNAVAILABLE"));
  setText("mmwaveRisk", riskComponentText(mmwave));

  const thermal = payload.thermal || {};
  const thermalAi = aiAt(thermal);
  setStatus("thermalStatus", thermal.state?.status);
  setText("thermalAi", thermalAi.available ? thermalAi.state : (thermalAi.error || "UNAVAILABLE"));
  const probabilities = thermalAi.metadata?.probabilities;
  const human = Array.isArray(probabilities) && probabilities.length === 3 ? probabilities[1] + probabilities[2] : null;
  setText("humanProbability", percent(human));
  const rawMin = thermalAi.metadata?.raw_minimum ?? valueAt(thermal, "minimum_raw");
  const rawMax = thermalAi.metadata?.raw_maximum ?? valueAt(thermal, "maximum_raw");
  setText("thermalRawRange", typeof rawMin === "number" && typeof rawMax === "number" ? `${rawMin} – ${rawMax}` : "—");
  setText("thermalRisk", riskComponentText(thermal));
  renderHeatmap(thermalAi.metadata?.heatmap_preview);

  const co2 = payload.co2 || {};
  const ppm = valueAt(co2, "ppm");
  setStatus("co2Status", co2.state?.status);
  setText("co2Value", number(ppm, 0));
  const co2Component = componentAt(co2);
  setText("co2State", co2Component.state || "—");
  setText("co2Risk", riskComponentText(co2));
  const co2Progress = typeof ppm === "number" ? Math.max(0, Math.min(100, ppm / 3000 * 100)) : 0;
  $("co2Track").style.width = `${co2Progress}%`;
  $("co2Track").style.background = ppm >= 2500 ? "var(--red)" : ppm >= 1000 ? "var(--yellow)" : "var(--cyan)";

  const pir = payload.pir || {};
  const motion = valueAt(pir, "motion");
  const pirComponent = componentAt(pir);
  setStatus("pirStatus", pir.state?.status);
  $("motionVisual").dataset.motion = typeof motion === "boolean" ? String(motion) : "unknown";
  setText("motionValue", typeof motion === "boolean" ? (motion ? "움직임 감지" : "움직임 없음") : "확인 불가");
  setText("pirRule", pirComponent.state || aiAt(pir).state || "—");
  const noMotion = pirComponent.metadata?.no_motion_seconds;
  setText("noMotionTime", typeof noMotion === "number" ? `${noMotion.toFixed(1)}초` : "—");
  setText("pirRisk", riskComponentText(pir));
}

function renderHeatmap(preview) {
  const canvas = $("thermalCanvas");
  const context = canvas.getContext("2d");
  context.fillStyle = "#071018";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const valid = preview && Number.isInteger(preview.width) && Number.isInteger(preview.height) &&
    Array.isArray(preview.values) && preview.values.length === preview.width * preview.height;
  if (!valid) {
    context.strokeStyle = "#162b37";
    for (let x = 0; x < canvas.width; x += 25) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, canvas.height); context.stroke(); }
    for (let y = 0; y < canvas.height; y += 25) { context.beginPath(); context.moveTo(0, y); context.lineTo(canvas.width, y); context.stroke(); }
    setText("heatmapNote", "FRAME UNAVAILABLE");
    canvas.setAttribute("aria-label", "Thermal frame을 사용할 수 없습니다");
    return;
  }
  const cellWidth = canvas.width / preview.width;
  const cellHeight = canvas.height / preview.height;
  preview.values.forEach((rawValue, index) => {
    const value = Math.max(0, Math.min(1, Number(rawValue) || 0));
    const hue = 245 - value * 245;
    const lightness = 24 + value * 34;
    context.fillStyle = `hsl(${hue} 88% ${lightness}%)`;
    context.fillRect((index % preview.width) * cellWidth, Math.floor(index / preview.width) * cellHeight, cellWidth + .5, cellHeight + .5);
  });
  setText("heatmapNote", `${preview.source_width}×${preview.source_height} RAW · NORMALIZED PREVIEW`);
  canvas.setAttribute("aria-label", `${preview.source_width} 곱하기 ${preview.source_height} Thermal frame 정규화 미리보기`);
}

function render(payload) {
  state.publication = payload;
  renderOverview(payload);
  renderSensors(payload);
  const now = Date.now();
  if (now - state.auxiliaryAt > 4500) {
    state.auxiliaryAt = now;
    refreshAuxiliary();
  }
}

async function refreshAuxiliary() {
  await Promise.allSettled([loadEvents(), loadHistory()]);
}

async function loadEvents() {
  const response = await fetch("/api/events?limit=8", { cache: "no-store" });
  if (!response.ok) throw new Error("events unavailable");
  const payload = await response.json();
  setText("storageBadge", (payload.persistence || "memory").toUpperCase());
  const list = $("eventList");
  list.replaceChildren();
  if (!payload.events?.length) {
    const item = document.createElement("li"); item.className = "empty-state"; item.textContent = "기록된 이벤트가 없습니다."; list.append(item); return;
  }
  payload.events.forEach((event) => {
    const item = document.createElement("li");
    const time = document.createElement("time"); time.textContent = formatTime(event.timestamp); time.dateTime = new Date(event.timestamp * 1000).toISOString();
    const content = document.createElement("div");
    const title = document.createElement("strong"); title.textContent = eventLabels[event.event_type] || event.event_type;
    const detail = document.createElement("span"); detail.textContent = eventDetail(event);
    content.append(title, detail); item.append(time, content); list.append(item);
  });
}

function eventDetail(event) {
  const details = event.details || {};
  if (details.sensor_id) return `${details.sensor_id} · ${details.from ?? "—"} → ${details.to ?? "—"}`;
  if ("from" in details || "to" in details) return `${details.from ?? "—"} → ${details.to ?? "—"}`;
  if (details.source) return `${details.source} · ${details.detail || "오류"}`;
  return Object.entries(details).map(([key, value]) => `${key}: ${value}`).join(" · ") || "상세 정보 없음";
}

async function loadHistory() {
  const response = await fetch("/api/history?limit=60", { cache: "no-store" });
  if (!response.ok) throw new Error("history unavailable");
  const payload = await response.json();
  drawTrend([...(payload.history || [])].reverse());
}

function drawTrend(history) {
  const canvas = $("trendCanvas");
  const context = canvas.getContext("2d");
  const width = canvas.width, height = canvas.height, pad = 34;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#203746"; context.lineWidth = 1;
  for (let row = 0; row <= 4; row++) {
    const y = pad + (height - pad * 2) * row / 4;
    context.beginPath(); context.moveTo(pad, y); context.lineTo(width - pad, y); context.stroke();
  }
  const usable = history.filter((item) => typeof item.timestamp === "number");
  $("trendEmpty").hidden = usable.length > 1;
  if (usable.length <= 1) { canvas.setAttribute("aria-label", "최근 기록이 부족합니다"); return; }
  const plot = (getter, max, color) => {
    context.beginPath(); context.strokeStyle = color; context.lineWidth = 3; context.lineJoin = "round";
    let started = false;
    usable.forEach((item, index) => {
      const value = getter(item);
      if (typeof value !== "number" || !Number.isFinite(value)) { started = false; return; }
      const x = pad + (width - pad * 2) * index / Math.max(1, usable.length - 1);
      const y = height - pad - Math.max(0, Math.min(1, value / max)) * (height - pad * 2);
      if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y);
    });
    context.stroke();
  };
  plot((item) => item.risk_score ?? item.risk?.risk_score, 100, "#ff5d6c");
  plot((item) => item.co2_ppm ?? item.state?.sensors?.co2?.values?.ppm, 3000, "#38d5e6");
  canvas.setAttribute("aria-label", `최근 ${usable.length}개 기록의 위험도와 CO₂ 추이`);
}

function setConnection(mode, label) {
  $("connectionBadge").dataset.state = mode;
  $("connectionBadge").querySelector("span").textContent = label;
}

function connect() {
  clearTimeout(state.reconnectTimer);
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${location.host}/ws`);
  state.socket = socket;
  setConnection("connecting", "연결 중");
  socket.addEventListener("open", () => { stopPolling(); setConnection("live", "실시간 연결"); });
  socket.addEventListener("message", (event) => {
    try { render(JSON.parse(event.data)); } catch (_error) { setConnection("polling", "데이터 오류"); }
  });
  socket.addEventListener("close", () => {
    if (state.socket !== socket) return;
    setConnection("polling", "Polling 전환"); startPolling();
    state.reconnectTimer = setTimeout(connect, 2500);
  });
  socket.addEventListener("error", () => socket.close());
}

function startPolling() {
  if (state.pollTimer) return;
  poll(); state.pollTimer = setInterval(poll, 2000);
}
function stopPolling() { clearInterval(state.pollTimer); state.pollTimer = null; }
async function poll() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error("status unavailable");
    render(await response.json()); setConnection("polling", "Polling 연결");
  } catch (_error) { setConnection("offline", "연결 끊김"); }
}

function formatTime(unixSeconds) {
  if (typeof unixSeconds !== "number" || !Number.isFinite(unixSeconds)) return "—";
  return new Intl.DateTimeFormat("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date(unixSeconds * 1000));
}

function tickClock() {
  const now = new Date(); $("currentTime").dateTime = now.toISOString(); $("currentTime").textContent = now.toLocaleTimeString("ko-KR", { hour12: false });
}

tickClock(); setInterval(tickClock, 1000); renderHeatmap(null); startPolling(); connect();
