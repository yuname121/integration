(() => {
  const standalone = location.protocol === "file:" || new URLSearchParams(location.search).has("standalone");
  let token = sessionStorage.getItem("safenest-token") || "";
  let eventStream = null;
  const loginForm = document.getElementById("adminLoginForm");
  const loginId = document.getElementById("adminLoginId");
  const loginPassword = document.getElementById("adminLoginPassword");
  const loginButton = document.getElementById("adminLoginButton");
  const loginError = document.getElementById("adminLoginError");

  const setAuthenticated = authenticated => {
    document.body.classList.toggle("auth-pending", !authenticated);
    if (!authenticated) setTimeout(() => loginPassword.focus(), 0);
  };

  const api = async (url, options={}) => {
    const response = await fetch(url, {
      ...options,
      headers:{
        ...(token ? { Authorization:`Bearer ${token}` } : {}),
        ...(options.body ? { "Content-Type":"application/json" } : {}),
        ...options.headers
      }
    });
    const data = response.status === 204 ? null : await response.json();
    if (response.status === 401) {
      token = "";
      sessionStorage.removeItem("safenest-token");
      setAuthenticated(false);
    }
    if (!response.ok) throw Error(data?.error || `HTTP ${response.status}`);
    return data;
  };

  const adapt = s => ({
    ...s,
    co2:s.reading?.co2 == null ? "-" : `${s.reading.co2} ppm`,
    temp:s.reading?.temperature == null ? "-" : `${s.reading.temperature}℃`,
    motion:s.reading?.motion ? "감지됨" : "미감지",
    last:s.lastSeen ? new Date(s.lastSeen).toLocaleTimeString("ko-KR") : "수신 대기"
  });

  function renderLiveSpace() {
    const space = currentSpace();
    if (!space || standalone) return;
    const reading = space.reading || {};
    const actual = { status:space.status, risk:space.risk, co2:space.co2, temp:space.temp, motion:space.motion, last:space.last };
    setDashboardMode(space.status in dashboardStates ? space.status : "offline", false);
    Object.assign(space, actual);
    if (space.status === "offline") return;

    const set = (id, value) => { const element=document.getElementById(id); if(element) element.textContent=value; };
    set("presenceText", reading.occupied ? "작업자 감지" : "인체 미감지");
    set("presenceSub", `Raspberry Pi · ${space.bridge?.deviceId || space.nodeId} · ${space.last}`);
    set("riskScore", `${space.risk ?? 0}/100`);
    set("riskBadge", stateMeta[space.status]?.label || space.status);
    set("breathValue", reading.breathRate == null ? "수신 대기" : `${Number(reading.breathRate).toFixed(1)} rpm`);
    set("breathFoot", reading.breathRate == null ? "MR60BHA2 데이터 없음" : "MR60BHA2 실시간 수신");
    set("heartValue", reading.heartRate == null ? "수신 대기" : `${Number(reading.heartRate).toFixed(1)} bpm`);
    set("heartFoot", reading.heartRate == null ? "MR60BHA2 데이터 없음" : "MR60BHA2 실시간 수신");
    const thermalMax = reading.thermal?.fresh
      ? reading.thermal.maxC
      : reading.bodyTemperature;
    set("tempLabel", "열화상 최고온도");
    set("tempValue", thermalMax == null ? "수신 대기" : `${Number(thermalMax).toFixed(1)}℃`);
    set("tempFoot", reading.thermal?.fresh ? "80×62 실시간 프레임" : "열화상 프레임 대기");
    set("co2Value", reading.co2 == null ? "수신 대기" : `${Math.round(reading.co2).toLocaleString()} ppm`);
    set("co2Foot", reading.co2 == null ? "SCD4x 데이터 없음" : reading.co2 >= 1500 ? "주의 기준 초과" : "SCD4x 실시간 수신");
    set("motionValue", reading.motion ? "감지됨" : "감지 안 됨");
    set("motionFoot", `PIR · 무움직임 ${reading.motionlessSeconds || 0}초`);
    set("fusionMmwave", reading.breathRate == null ? "수신 대기" : "호흡·심박 수신");
    set("fusionThermal", reading.thermal?.fresh ? "실시간 프레임" : "프레임 대기");
    set("fusionMotion", reading.motion ? "최근 감지" : "미감지");
    set("fusionCo2", reading.co2 == null ? "수신 대기" : reading.co2 >= 1500 ? "환기 필요" : "정상");
    set("confidenceValue", space.bridge?.fresh ? "LIVE" : "대기");
    set("confidenceText", "ESP32 → Raspberry Pi → SafeNest Web");
  }

  async function reload() {
    const [remoteSpaces, remoteEvents] = await Promise.all([api("/api/spaces"), api("/api/portal/events")]);
    spaces = remoteSpaces.map(adapt);
    events = remoteEvents.map(e => ({ ...e, time:new Date(e.time).toLocaleString("sv-SE") }));
    if (!getSpace(currentSpaceId)) currentSpaceId = spaces[0]?.id;
    renderCurrentRoom();
    renderRecentEvents();
    renderLiveSpace();
    renderSpaces();
    renderSettings();
    updateGlobalConnection();
  }

  function connectStream() {
    if (eventStream) eventStream.close();
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    eventStream = new WebSocket(`${protocol}//${location.host}/ws`);
    eventStream.addEventListener("message", () => reload().catch(error => showToast(error.message)));
    eventStream.addEventListener("close", () => {
      if (token) setTimeout(connectStream, 2500);
    });
  }

  loginForm.addEventListener("submit", async event => {
    event.preventDefault();
    loginError.textContent = "";
    loginButton.disabled = true;
    loginButton.textContent = "로그인 중...";
    try {
      if (standalone) {
        if (loginId.value.trim() !== "admin" || loginPassword.value !== "SafeNest123!")
          throw Error("아이디 또는 비밀번호가 올바르지 않습니다.");
        token = "standalone-preview";
        sessionStorage.setItem("safenest-token", token);
        setAuthenticated(true);
        loginPassword.value = "";
        return;
      }
      const response = await fetch("/api/auth/login", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body:JSON.stringify({ id:loginId.value.trim(), password:loginPassword.value })
      });
      const data = await response.json();
      if (!response.ok) throw Error(data.error || "로그인할 수 없습니다.");
      token = data.token;
      sessionStorage.setItem("safenest-token", token);
      await reload();
      setAuthenticated(true);
      loginPassword.value = "";
      connectStream();
    } catch (error) {
      token = "";
      sessionStorage.removeItem("safenest-token");
      loginError.textContent = error.message;
      setAuthenticated(false);
    } finally {
      loginButton.disabled = false;
      loginButton.textContent = "로그인";
    }
  });

  if (!standalone) document.getElementById("connectionForm").addEventListener("submit", async event => {
    event.preventDefault();
    event.stopImmediatePropagation();
    try {
      await api("/api/spaces", { method:"POST", body:JSON.stringify({
        name:document.getElementById("newSpaceName").value.trim(),
        nodeId:document.getElementById("newNodeId").value.trim(),
        host:document.getElementById("newHost").value.trim(),
        port:document.getElementById("newPort").value.trim()
      }) });
      event.target.reset();
      document.getElementById("newPort").value = "8000";
      document.getElementById("healthPath").value = "/health";
      showToast("공간과 QR URL을 등록했습니다.");
      await reload();
    } catch (error) { showToast(error.message); }
  }, true);

  if (!standalone) removeSpace = async id => {
    const item = getSpace(id);
    if (!item || !confirm(`${item.name} 연결 정보를 삭제할까요?`)) return;
    try {
      await api(`/api/spaces/${encodeURIComponent(id)}`, { method:"DELETE" });
      await reload();
      showToast("공간 연결 정보를 삭제했습니다.");
    } catch (error) { showToast(error.message); }
  };

  if (!standalone) document.getElementById("renameSpaceBtn").addEventListener("click", async event => {
    event.preventDefault();
    event.stopImmediatePropagation();
    const id = document.getElementById("renameSpaceSelect").value;
    const name = document.getElementById("renameSpaceInput").value.trim();
    if (!id || !name) return showToast("변경할 공간과 이름을 확인하세요.");
    try {
      await api(`/api/spaces/${encodeURIComponent(id)}`, {
        method:"PATCH",
        body:JSON.stringify({ name })
      });
      await reload();
      showToast("공간 이름을 변경했습니다.");
    } catch (error) { showToast(error.message); }
  }, true);

  const actions = document.querySelector(".header-actions");
  const logout = document.createElement("button");
  logout.className = "secondary-button";
  logout.textContent = "로그아웃";
  logout.onclick = () => {
    token = "";
    sessionStorage.removeItem("safenest-token");
    if (eventStream) eventStream.close();
    setAuthenticated(false);
  };
  actions.append(logout);

  setInterval(() => {
    if (token && !standalone) reload().catch(() => {});
  }, 1000);

  if (standalone && token === "standalone-preview") {
    setAuthenticated(true);
  } else if (token) {
    reload().then(() => { setAuthenticated(true); connectStream(); }).catch(error => {
      loginError.textContent = error.message;
      setAuthenticated(false);
    });
  } else {
    setAuthenticated(false);
  }
})();


