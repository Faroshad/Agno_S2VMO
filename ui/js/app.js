'use strict';
/* ══════════════════════════════════════════════════════
   app.js — Application state, WebSocket, MQTT, simulation,
            Neo4j, Vision, Chat, Icons, and boot sequence
   Depends on: utils.js, nodeflow.js, chart-view.js, viewer.mjs (imported at boot)
══════════════════════════════════════════════════════ */

/* ── Global shared state ──────────────────────────── */
const API = '';
let ws = null;
let simRunning        = false;
let sensorsAssigned   = false;  /* true once user assigns voxel positions */
let voxelData         = null;   /* shared with viewer.js */
let stressData        = null;   /* shared with viewer.js */
let stressThreshold   = 0.5;    /* shared with viewer.js */
let state_femModelReady = false; /* true when all FEM model files are present */
/* ══════════════════════════════════════════════════════
   LUCIDE ICONS
══════════════════════════════════════════════════════ */
function toCamel(s) {
  return s.split('-').map(p => p[0].toUpperCase() + p.slice(1)).join('');
}

function setIcon(containerId, name, size = 14) {
  const el = document.getElementById(containerId);
  if (!el || typeof lucide === 'undefined') return;
  el.innerHTML = '';
  const svg = lucide.createElement(lucide[toCamel(name)]);
  svg.setAttribute('width', size);
  svg.setAttribute('height', size);
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.75');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  el.appendChild(svg);
}

function initIcons() {
  // top bar
  setIcon('ico-mqtt',    'radio',               14);
  setIcon('ico-neo4j',   'database',            14);
  setIcon('ico-sim',     'activity',            14);
  setIcon('ico-fem',     'cpu',                 14);
  setIcon('ico-bot',          'bot',             16);
  setIcon('ico-chat-clear',   'trash-2',         12);
  setIcon('ico-send',         'send',            14);
  setIcon('ico-ctx-sim',      'activity',        10);
  setIcon('ico-ctx-mqtt',     'radio',           10);
  // floating panels
  setIcon('ico-assign',  'map-pin',             14);
  setIcon('ico-alert',   'alert-circle',        14);
  setIcon('ico-legend',  'layers',              14);
  // 3D viewer controls
  setIcon('ico-reset',   'rotate-ccw',          14);
  setIcon('ico-wire',    'box',                 14);
  setIcon('ico-pins',    'map-pin',             14);
  // node icons
  setIcon('ni-mqtt',     'radio',               14);
  setIcon('ni-sens',     'sliders-horizontal',  14);
  setIcon('ni-fem',      'cpu',                 14);
  setIcon('ni-neo4j',    'database',            14);
  setIcon('ni-chart-mpu',  'activity',       14);
  setIcon('ni-chart-sg',   'trending-up',    14);
  setIcon('ni-chart-fem',  'bar-chart-2',    14);
  setIcon('ni-stats',      'zap',            14);
  setIcon('ni-log',        'terminal',       14);
}

/* ══════════════════════════════════════════════════════
   WEBSOCKET
══════════════════════════════════════════════════════ */
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen    = () => { setStatus('sb-ws-state', 'WS: connected'); log('WebSocket connected', 'ok'); ping(); };
  ws.onclose   = () => { setStatus('sb-ws-state', 'WS: reconnecting…'); setTimeout(connectWS, 3000); };
  ws.onerror   = () => {};
  ws.onmessage = e  => {
    // Server sends plain-text "pong" for keep-alive — ignore it, not JSON
    if (e.data === 'pong') return;
    try { handleWS(JSON.parse(e.data)); } catch(err) { /* ignore malformed frames */ }
  };
}

function ping() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send('ping');
    setTimeout(ping, 30000);
  }
}

function handleWS(msg) {
  const { type, data } = msg;
  switch (type) {
    case 'state_snapshot':    applySnapshot(data);                               break;
    case 'mqtt_status':         updateMQTTStatus(data.connected);                break;
    case 'mqtt_message':        onMQTTMessage(data);                             break;
    case 'mqtt_sensor_update':  onLiveSensorUpdate(data);                        break;
    case 'sim_status':          updateSimStatus(data.running);                   break;
    case 'sim_cycle':         onSimCycle(data);                                  break;
    case 'fem_cycle_start':   onFemCycleStart(data);                             break;
    case 'sim_error':         log(`Sim error: ${data.error}`, 'err');            break;
    case 'sensors_assigned':  onSensorsAssigned(data);                           break;
    case 'sensors_confirmed': log(`${data.count} sensors confirmed in DB`, 'ok'); break;
    case 'neo4j_updated':     applyNeo4jCounts(data);                            break;
    case 'neo4j_status':      updateNeo4jDot(!!data.connected);                break;
    case 'fem_ready':         applyFemReady(data);                               break;
    case 'agent_ready':       applyAgentReady(data);                             break;
  }
}

function applySnapshot(d) {
  updateMQTTStatus(d.mqtt_connected);
  updateSimStatus(d.sim_running);
  updateNeo4jDot(d.neo4j_connected);
  if (d.sensor_history) updateChart(d.sensor_history);
  ge('stat-cycle').textContent = d.sim_cycle || 0;
  ge('sb-cycle').textContent   = d.sim_cycle || 0;
  /* Restore interval slider if included in snapshot */
  if (d.sim_interval) {
    const slider = ge('sim-interval-slider');
    const label  = ge('sim-interval-val');
    if (slider) slider.value = d.sim_interval;
    if (label)  label.textContent = `${d.sim_interval} s`;
  }
  /* Restore FEM model + agent dots */
  if (d.fem_model_ready !== undefined) applyFemReady({ ready: d.fem_model_ready });
  if (d.agent_ready     !== undefined) applyAgentReady({ ready: d.agent_ready, error: d.agent_error });
}

/* ── FEM model readiness ───────────────────────────────────────────────────── */
function applyFemReady(data) {
  state_femModelReady = !!data.ready;
  /* dot-fem (topbar) and nd-fem (node): green=running, amber=ready+idle, red=missing */
  if (!simRunning) {
    const s = state_femModelReady ? 'warn' : '';
    setDot('dot-fem', s);
    setDot('nd-fem',  s);
  }
  const txt = state_femModelReady ? 'ready · idle' : 'model files missing';
  ge('fem-status-val').textContent = txt;
  if (!state_femModelReady && data.missing?.length)
    log(`FEM: missing files — ${data.missing.join(', ')}`, 'warn');
}

/* ── GraphRAG agent readiness ─────────────────────────────────────────────── */
function applyAgentReady(data) {
  const ready = data.ready;
  const err   = data.error || '';
  setDot('dot-agent', ready ? 'on' : (err ? '' : 'warn'));
  if (!ready && err) log(`GraphRAG agent: ${err}`, 'warn');
}

/* ══════════════════════════════════════════════════════
   SENSOR ASSIGNMENT
══════════════════════════════════════════════════════ */
function _intVal(id) {
  const v = parseInt(ge(id)?.value, 10);
  return isNaN(v) ? null : v;
}

async function assignSensors() {
  const sensors = [
    { id: 'S1', grid_i: _intVal('s1-i'), grid_j: _intVal('s1-j'), grid_k: _intVal('s1-k'), sensor_type: 'MPU' },
    { id: 'S2', grid_i: _intVal('s2-i'), grid_j: _intVal('s2-j'), grid_k: _intVal('s2-k'), sensor_type: 'MPU' },
    { id: 'S3', grid_i: _intVal('s3-i'), grid_j: _intVal('s3-j'), grid_k: _intVal('s3-k'), sensor_type: 'SG'  },
    { id: 'S4', grid_i: _intVal('s4-i'), grid_j: _intVal('s4-j'), grid_k: _intVal('s4-k'), sensor_type: 'SG'  },
  ];

  const invalid = sensors.filter(s => s.grid_i === null || s.grid_j === null || s.grid_k === null);
  if (invalid.length > 0) {
    log(`Fill all i/j/k values for: ${invalid.map(s => s.id).join(', ')}`, 'err');
    return;
  }

  const btn = ge('btn-assign-sensors');
  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    const r = await apiFetch('/api/sensors/assign', 'POST', sensors);
    if (r.sensors_assigned) {
      onSensorsAssigned({ positions: r.positions });
      log('Sensors assigned — MQTT connect is now unlocked', 'ok');
    }
  } catch (e) {
    log(`Assign failed: ${e.message}`, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Reassign';
  }
}

function onSensorsAssigned(data) {
  sensorsAssigned = true;
  const pos = data.positions || {};

  /* Update floating panel cards */
  ['S1','S2','S3','S4'].forEach((id, i) => {
    const coords = pos[id];
    const posEl  = ge(`sc-pos-s${i + 1}`);
    if (posEl) {
      if (coords) {
        posEl.textContent = `(${coords[0]}, ${coords[1]}, ${coords[2]})`;
        posEl.classList.remove('unset');
      } else {
        posEl.textContent = 'not assigned';
        posEl.classList.add('unset');
      }
    }
    /* Do NOT pre-fill inputs — HTML defaults are the canonical source of truth */
  });

  /* Update Sensor Config node in flow */
  const labels = ['S1 MPU','S2 MPU','S3 SG','S4 SG'];
  ['S1','S2','S3','S4'].forEach((id, i) => {
    const el  = ge(`nd-pos-s${i + 1}`);
    const coords = pos[id];
    if (el) {
      el.textContent = coords
        ? `${labels[i]}  (${coords[0]}, ${coords[1]}, ${coords[2]})`
        : `${labels[i]}  — not set`;
      el.style.color = coords ? 'var(--accent)' : 'var(--txt2)';
    }
  });
  const srcEl = ge('nd-sens-src');
  if (srcEl) { srcEl.textContent = 'MQTT · sensors assigned'; srcEl.style.color = 'var(--accent)'; }

  /* Set sensor node status dot to on */
  setDot('nd-sens', 'on');

  ge('btn-assign-sensors').textContent = 'Reassign';
  ge('btn-assign-sensors').classList.add('assigned');

  /* Update voxel geometry (re-fetches so new sensor voxels turn blue when sim runs) */
  if (typeof loadVoxelGeometry === 'function') loadVoxelGeometry();
}

/* ── Status helpers ── */
function updateMQTTStatus(connected) {
  setDot('dot-mqtt', connected ? 'on' : '');
  setDot('nd-mqtt',  connected ? 'on' : '');
  ge('mqtt-toggle-btn').textContent = connected ? 'Disconnect' : 'Connect';
  ge('mqtt-toggle-btn').className   = `node-btn ${connected ? 'stop' : ''}`;
  ge('mqtt-toggle-btn').disabled    = false; /* MQTT can connect independently of sensor assignment */
  const badge = ge('mqtt-live-badge');
  if (badge) badge.className = connected ? 'live' : '';
  if (connected) {
    wireActive('wire-mqtt-sens', true);
  }
  /* Simulation runs independently of MQTT — always keep button enabled */
  ge('btn-run-sim').disabled = false;
}

function updateSimStatus(running) {
  simRunning = running;
  updateChatContext();
  setDot('dot-sim',  running ? 'on' : '');
  setDot('nd-stats', running ? 'on' : '');
  ge('btn-run-sim').disabled    = false;
  ge('btn-run-sim').textContent = running ? 'Stop Simulation' : 'Run Simulation';
  /* FEM dot: green=running, amber=model ready+idle, red=missing */
  const femDotState = running ? 'on' : (state_femModelReady ? 'warn' : '');
  setDot('dot-fem', femDotState);
  setDot('nd-fem',  femDotState);
  ge('fem-status-val').textContent = running ? 'running…' : (state_femModelReady ? 'ready · idle' : 'model files missing');
  wireActive('wire-sens-fem',  running);
  wireActive('wire-fem-neo4j', running);
  wireActive('wire-fem-sg',    running);
  wireActive('wire-fem-stats', running);
}

/* Called on fem_cycle_start — briefly pulse the FEM node dot bright green */
function onFemCycleStart(data) {
  const dot = ge('nd-fem');
  if (!dot) return;
  dot.classList.add('pulse-active');
  setTimeout(() => dot.classList.remove('pulse-active'), 600);
  ge('fem-status-val').textContent = `computing cycle ${data.cycle}…`;
}

/* Update Neo4j node with fresh counts from server */
function applyNeo4jCounts(r) {
  if (!r) return;
  /* Explicit false / error → offline. Otherwise treat as online (avoids !undefined → skip). */
  if (r.connected === false || r.error) {
    updateNeo4jDot(false);
    return;
  }
  updateNeo4jDot(true);
  const set = (id, v) => { const el = ge(id); if (el) el.textContent = v; };
  set('neo4j-voxels',      r.voxel_count   ?? '—');
  set('neo4j-analyses',    r.fem_analyses  ?? '—');
  set('neo4j-streams',     r.sensor_streams ?? '—');
  set('neo4j-total-nodes', r.total_nodes   ?? '—');
  set('neo4j-total-rels',  r.total_rels    ?? '—');
  set('neo4j-updated-at',  r.updated_at    ?? '—');
  /* Also flash the Neo4j dot to show update activity */
  const dot = ge('nd-neo4j');
  if (dot) {
    dot.classList.add('pulse-active');
    setTimeout(() => dot.classList.remove('pulse-active'), 600);
  }
  /* Mirror to status-bar / dome panel */
  const dbVox = ge('db-voxels');
  const dbFem = ge('db-fem');
  if (dbVox) dbVox.textContent = `Voxels: ${r.voxel_count ?? '—'}`;
  if (dbFem) dbFem.textContent = `FEM runs: ${r.fem_analyses ?? '—'}`;
}

async function onSimIntervalChange(val) {
  const v = parseInt(val, 10);
  const label = ge('sim-interval-val');
  if (label) label.textContent = `${v} s`;
  try {
    await apiFetch('/api/sim/interval', 'PUT', { interval_s: v });
  } catch (e) { /* ignore */ }
}

function updateNeo4jDot(connected) {
  setDot('dot-neo4j', connected ? 'on' : '');
  setDot('nd-neo4j',  connected ? 'on' : '');
}

/* ══════════════════════════════════════════════════════
   MQTT
══════════════════════════════════════════════════════ */
async function toggleMQTT() {
  const connected = ge('mqtt-toggle-btn').textContent.includes('Disconnect');
  if (connected) {
    await apiFetch('/api/mqtt/stop', 'POST');
    log('MQTT disconnected', 'warn');
  } else {
    /* Reset heatmap rolling baseline so old readings don't bleed into new session */
    if (typeof resetHeatmapHistory === 'function') resetHeatmapHistory();
    await apiFetch('/api/mqtt/start', 'POST', {
      broker: ge('mqtt-broker').value,
      port: 1883,
      topic: ge('mqtt-topic').value,
    });
    log(`MQTT connecting → ${ge('mqtt-broker').value}`, 'info');
  }
}

function onMQTTMessage(data) {
  const cnt = ge('mqtt-count');
  cnt.textContent = `${(parseInt(cnt.textContent) || 0) + 1} messages`;
  setDot('nd-chart-mpu', 'on');
  setDot('nd-chart-sg',  'on');
  wireActive('wire-mqtt-sens', true);
  wireActive('wire-mqtt-mpu',  true);
  wireActive('wire-fem-sg',    false); /* FEM not triggered by raw MQTT message */
  /* Sensor values are shown via onLiveSensorUpdate which fires on mqtt_sensor_update */
}

/* Called when the server has parsed the physical-device array payload */
function onLiveSensorUpdate(data) {
  const readings = data.readings || [];       // [S1, S2, S3, S4] strain_uE
  const rich     = data.rich     || {};
  const ts       = data.ts       || new Date().toISOString();
  updateChatContext();

  /* Status bar sensor display */
  const sensorLabel = ge('mqtt-sensor-val');
  if (sensorLabel && readings.length === 4) {
    sensorLabel.textContent =
      `S3:${readings[2].toFixed(0)} S4:${readings[3].toFixed(0)} μE`;
  }

  /* Push to BOTH specialised charts */
  if (typeof pushMpuData === 'function') pushMpuData(ts, rich);
  if (typeof pushSgData  === 'function') pushSgData(ts, rich, readings);

  /* Drive sensor heatmap spheres with live readings */
  if (typeof updateSensorHeatmap === 'function') updateSensorHeatmap(readings);

  /* Status bar readings */
  ['s1','s2','s3','s4'].forEach((k, i) => {
    const el = ge(`sb-${k}`);
    if (el && readings[i] !== undefined) el.textContent = readings[i].toFixed(1);
  });

  /* Log summary */
  const r1 = rich.S1 || {};
  const accMag = Math.sqrt(
    (r1.acc_g?.x || 0) ** 2 + (r1.acc_g?.y || 0) ** 2 + (r1.acc_g?.z || 0) ** 2
  ).toFixed(3);
  log(`MQTT ← strain: [${readings.map(v => v.toFixed(0)).join(', ')}] μE  |  |acc₁|: ${accMag} g`, 'info');

  setDot('nd-chart-mpu', 'on');
  setDot('nd-chart-sg',  'on');
  wireActive('wire-mqtt-sens', true);
}

/* ══════════════════════════════════════════════════════
   SIMULATION
══════════════════════════════════════════════════════ */
async function toggleSim() {
  if (simRunning) {
    await apiFetch('/api/sim/stop', 'POST');
    log('Simulation stopped', 'warn');
  } else {
    await apiFetch('/api/sim/start', 'POST');
    log('Simulation started (waiting for MQTT data)', 'ok');
  }
}

async function singleStep() {
  log('Running single FEM cycle (MQTT sensor data)…', 'info');
  try {
    const r = await apiFetch('/api/sim/step', 'POST');
    log(`Cycle ${r.cycle}: max ${fmt(r.max_stress_pa)} Pa`, 'ok');
    onSimCycle(r);
    refreshVoxels();
  } catch (e) {
    log(`Step failed: ${e.message}`, 'err');
  }
}

function onSimCycle(d) {
  ge('stat-cycle').textContent = d.cycle || 0;
  ge('sb-cycle').textContent   = d.cycle || 0;
  const avg = d.avg_stress_pa || d.avg_stress || 0;
  const max = d.max_stress_pa || d.max_stress || 0;
  ge('stat-avg-stress').textContent = fmt(avg) + ' Pa';
  ge('stat-max-stress').textContent = fmt(max) + ' Pa';
  ge('stat-max-stress').className   = `val ${max > 3000 ? 'crit' : max > 2000 ? 'warn' : ''}`;
  ge('sb-stress').textContent       = fmt(avg);
  ge('fem-cycle-val').textContent   = `${d.cycle} / ${fmt(max)} Pa`;
  const srcLabel = d.source ? (d.source.startsWith('MQTT') ? '📡 MQTT' : '📂 defaults') : '';
  // Show FEM smoothing mode badge: NN (full compute), EMA (blended), GATE (reused)
  const modeMap = {
    'nn_full':        '🔮 NN',
    'nn_first_cycle': '🔮 NN',
    'ema_blend':      `🔀 EMA α=${(d.fem_alpha||0).toFixed(2)}`,
    'gate_reuse':     '⏸ GATE',
  };
  const modeLabel = modeMap[d.fem_mode] || '🔮 NN';
  const deltaLbl  = (d.sensor_delta_uE !== undefined && d.sensor_delta_uE !== null)
                      ? ` Δ${d.sensor_delta_uE.toFixed(1)}μE` : '';
  const mqttLbl   = d.mqtt_samples ? ` n=${d.mqtt_samples}` : '';
  ge('fem-status-val').textContent  = `cycle ${d.cycle} · ${srcLabel} · ${modeLabel}${deltaLbl}${mqttLbl}`;
  if (d.voxels_updated) ge('stat-voxels').textContent = d.voxels_updated.toLocaleString();

  const readings = d.sensor_readings || [];
  ['s1','s2','s3','s4'].forEach((k, i) => {
    if (readings[i] !== undefined) ge(`sb-${k}`).textContent = readings[i].toFixed(1);
  });
  if (readings.length === 4) {
    const ts = d.timestamp || new Date().toISOString();
    if (typeof pushSgData === 'function') pushSgData(ts, null, readings);
    /* Drive sensor heatmap spheres with sim-cycle readings */
    if (typeof updateSensorHeatmap === 'function') updateSensorHeatmap(readings);
  }

  /* Flash FEM dot green on each cycle completion */
  setDot('nd-fem',          'on');
  setDot('nd-chart-fem',    'on');
  wireActive('wire-fem-neo4j',    true);
  wireActive('wire-fem-sg',       true);
  wireActive('wire-fem-stats',    true);
  wireActive('wire-fem-femcycle', true);

  /* Push to FEM Cycles chart */
  if (typeof pushFemCycleData === 'function') pushFemCycleData(d);

  log(`FEM cycle ${d.cycle}: max ${fmt(max)} Pa  |  voxels: ${d.voxels_updated ?? '—'}  |  ${d.duration_s ?? '—'} s  |  ${modeLabel}${deltaLbl}`, 'ok');

  refreshVoxels();
  /* neo4j_updated is broadcast by server after each cycle automatically */
}

/* ══════════════════════════════════════════════════════
   NEO4J
══════════════════════════════════════════════════════ */
async function refreshNeo4j() {
  try {
    const r = await apiFetch('/api/neo4j/status');
    applyNeo4jCounts(r);
  } catch (e) {
    updateNeo4jDot(false);
  }
}

async function initNeo4j() {
  showModal('Initialize Database',
    `<p style="font-size:12.5px;color:var(--txt1);line-height:1.7;">
     Run in your terminal:<br/>
     <code style="display:block;background:var(--bg-d);box-shadow:inset 2px 2px 6px rgba(160,160,160,.4);
       padding:10px 12px;border-radius:8px;margin-top:8px;
       font-family:'Roboto Mono',monospace;font-size:11px;color:var(--txt0);">
       python scripts/initialize_neo4j.py</code>
     Then click <strong>Refresh</strong> on the Neo4j node.</p>`);
}

async function fetchAlerts() {
  try {
    const alerts = await apiFetch('/api/neo4j/alerts');
    ge('alert-count').textContent = alerts.length;
    if (alerts.length) {
      ge('msgs-badge').style.display = 'inline';
      ge('msgs-badge').textContent   = `${alerts.length} alert${alerts.length > 1 ? 's' : ''}`;
    }
    ge('alerts-list').innerHTML = alerts.map(a =>
      `<div class="alert-item ${a.severity === 'critical' ? 'crit' : ''}">
         ${alertIconSvg(a.severity)}
         <span>${(a.severity || '').toUpperCase()} ${fmt(a.stress_pa)} Pa @ (${a.gi},${a.gj},${a.gk})</span>
       </div>`
    ).join('');
  } catch (e) { /* offline */ }
}

/* Vision pipeline removed — sensor positions are mapped from physical dome image */

/* ══════════════════════════════════════════════════════
   CHAT
══════════════════════════════════════════════════════ */

/* ── marked.js config (set once on load) ── */
function _initMarked() {
  if (typeof marked === 'undefined') return;

  /* Custom renderer for tighter control over output HTML */
  const renderer = new marked.Renderer();

  /* Wrap tables in a scrollable container so wide tables don't overflow */
  renderer.table = (header, body) =>
    `<div class="md-table-wrap"><table><thead>${header}</thead><tbody>${body}</tbody></table></div>`;

  /* Code blocks: preserve language class for potential future syntax highlight */
  renderer.code = (code, lang) => {
    const cls = lang ? ` class="language-${lang}"` : '';
    return `<pre><code${cls}>${code}</code></pre>`;
  };

  marked.setOptions({
    renderer,
    breaks:    true,    // single newline → <br>
    gfm:       true,    // GitHub-flavoured markdown (tables, strikethrough, etc.)
    headerIds: false,   // no anchor ids (cleaner DOM)
    mangle:    false,
  });
}

/* ── Context strip: reflects live sim/MQTT state at a glance ── */
function updateChatContext() {
  const cycle = ge('stat-cycle')?.textContent || '—';
  const simEl = ge('ctx-sim');
  const mqttEl = ge('ctx-mqtt');
  const rdgEl = ge('ctx-readings');
  if (!simEl) return;

  simEl.textContent  = simRunning ? `cycle ${cycle} · running` : `cycle ${cycle} · stopped`;
  mqttEl.textContent = (ge('dot-mqtt')?.classList.contains('on')) ? 'MQTT live' : 'MQTT offline';

  /* Latest sensor readings from status bar spans */
  const s = ['s1','s2','s3','s4']
    .map((k, i) => `S${i+1}:${ge('sb-' + k)?.textContent || '—'}`)
    .join('  ');
  if (rdgEl) rdgEl.textContent = s;
}

/* ── Send a chat message ── */
async function sendChat() {
  const input = ge('chat-input');
  const msg   = input.value.trim();
  if (!msg) return;

  input.value = '';
  autoResizeChatInput(input);
  appendBubble('user', msg);

  const typing = ge('chat-typing');
  typing.style.display = 'flex';
  scrollChat();
  ge('chat-send').disabled = true;
  setDot('dot-agent', 'warn');

  /* Cycle through thinking labels to give feedback during long queries */
  const _labels = [
    'Querying knowledge graph…',
    'Running Cypher query…',
    'Analysing FEM data…',
    'Reading sensor history…',
    'Generating response…',
  ];
  let _li = 0;
  const _lbl = ge('chat-thinking-lbl');
  const _labelInterval = setInterval(() => {
    if (_lbl) _lbl.textContent = _labels[(_li = (_li + 1) % _labels.length)];
  }, 2800);

  try {
    const r = await apiFetch('/api/chat', 'POST', { message: msg });
    const text = r.answer || r.response || JSON.stringify(r);
    appendBubble('assistant', text, r.voxels || null);
  } catch (e) {
    appendBubble('assistant', `Error: ${e.message}`);
  } finally {
    clearInterval(_labelInterval);
    if (_lbl) _lbl.textContent = 'Querying knowledge graph…';
    typing.style.display = 'none';
    ge('chat-send').disabled = false;
    setDot('dot-agent', 'on');
    scrollChat();
  }
}

/* ── Append a message bubble ── */
function appendBubble(role, text, voxels) {
  const div = document.createElement('div');
  div.className = `chat-bubble ${role}`;

  if (role === 'assistant' && typeof marked !== 'undefined') {
    /* Sanitise: strip <script> tags before rendering markdown */
    const safe = text.replace(/<script[\s\S]*?<\/script>/gi, '');
    div.innerHTML = marked.parse(safe);
    /* Make all links open in new tab */
    div.querySelectorAll('a').forEach(a => { a.target = '_blank'; a.rel = 'noopener'; });
  } else {
    div.textContent = text;
  }

  ge('chat-msgs').appendChild(div);

  /* Pin referenced voxels in the 3D model for any voxel list source */
  if (role === 'assistant' && typeof setVoxelPins === 'function') {
    const pins = (voxels && voxels.length > 0)
      ? voxels
      : (typeof _extractVoxelPins === 'function' ? _extractVoxelPins(text) : []);
    if (pins.length > 0) {
      setVoxelPins(pins);
      log(`Pinned ${pins.length} voxel${pins.length > 1 ? 's' : ''} from AI response`, 'info');
    }
  }

  scrollChat();
}

/* ── Suggestion chip click ── */
function chipSend(btn) {
  const msg = btn.textContent.trim();
  const input = ge('chat-input');
  input.value = msg;
  autoResizeChatInput(input);
  sendChat();
}

/* ── Auto-grow textarea ── */
function autoResizeChatInput(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

/* ── Clear conversation ── */
async function chatClear() {
  try { await apiFetch('/api/chat/clear', 'POST'); } catch (_) { /* offline — clear UI anyway */ }
  ge('chat-msgs').innerHTML =
    `<div class="chat-bubble system">History cleared. Ask a new question.</div>`;
  scrollChat();
}

/* ── Load server-side chat history on page load ── */
async function loadChatHistory() {
  try {
    const history = await apiFetch('/api/chat/history');
    if (!Array.isArray(history) || history.length === 0) return;
    /* Prepend before the welcome message or replace it */
    const msgs = ge('chat-msgs');
    msgs.innerHTML = '';   // remove default system bubble
    for (const m of history) {
      if (m.role === 'user' || m.role === 'assistant') {
        appendBubble(m.role, m.content);
      }
    }
    /* Re-add separator after history */
    const sep = document.createElement('div');
    sep.className = 'chat-bubble system';
    sep.textContent = '— conversation restored —';
    msgs.appendChild(sep);
    scrollChat();
  } catch (_) { /* history endpoint not ready */ }
}

function chatKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
}

/* ══════════════════════════════════════════════════════
   VOXEL PIN EXTRACTION
   Parses an AI response string for Neo4j voxel references
   and calls setVoxelPins() in viewer.js.

   Recognised patterns (as produced by Cypher queries):
     • grid coords:   (i=10, j=20, k=5)  /  grid_i:10  /  10,20,5
     • world coords:  x:-1.23, y:0.45, z:2.10
     • mixed in text: "(3.9, -3.7, 0.4)"
══════════════════════════════════════════════════════ */
function _extractVoxelPins(text) {
  const voxels = [];
  const seen   = new Set();

  function _add(obj) {
    const key = JSON.stringify(obj);
    if (!seen.has(key)) { seen.add(key); voxels.push(obj); }
  }

  // 1. grid_i / grid_j / grid_k  (e.g. "grid_i: 10, grid_j: 20, grid_k: 5")
  const reGrid = /grid_i[:\s]+(-?\d+)[,\s]+grid_j[:\s]+(-?\d+)[,\s]+grid_k[:\s]+(-?\d+)/gi;
  let m;
  while ((m = reGrid.exec(text)) !== null) {
    _add({ gi: +m[1], gj: +m[2], gk: +m[3], label: `(${m[1]},${m[2]},${m[3]})` });
  }

  // 2. i=N, j=N, k=N notation (short form sometimes in logs)
  const reIJK = /\bi\s*=\s*(-?\d+)[,\s]+j\s*=\s*(-?\d+)[,\s]+k\s*=\s*(-?\d+)/gi;
  while ((m = reIJK.exec(text)) !== null) {
    _add({ gi: +m[1], gj: +m[2], gk: +m[3], label: `(${m[1]},${m[2]},${m[3]})` });
  }

  // 3. World coords:  x:-1.23, y:0.45, z:2.10  (from RETURN v.x, v.y, v.z)
  //    also handles x= or x =  assignment-style formatting
  const reXYZ = /x[\s:=]+([-\d.]+)[,;\s]+y[\s:=]+([-\d.]+)[,;\s]+z[\s:=]+([-\d.]+)/gi;
  while ((m = reXYZ.exec(text)) !== null) {
    _add({ x: +m[1], y: +m[2], z: +m[3], label: `(${(+m[1]).toFixed(2)},${(+m[2]).toFixed(2)},${(+m[3]).toFixed(2)})` });
  }

  // 4. Parenthesised triples like "(3.9, -3.7, 0.4)" or "(10, 20, 5)"
  // Only capture if they look like coordinate triples (3 numeric values)
  const reTuple = /\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)/g;
  while ((m = reTuple.exec(text)) !== null) {
    const a = +m[1], b = +m[2], c = +m[3];
    // Heuristic: if all three are small integers → likely grid coords; else world coords
    const isGrid = Number.isInteger(a) && Number.isInteger(b) && Number.isInteger(c)
                   && Math.abs(a) < 500 && Math.abs(b) < 500 && Math.abs(c) < 500;
    if (isGrid) {
      _add({ gi: a, gj: b, gk: c, label: `(${a},${b},${c})` });
    } else {
      _add({ x: a, y: b, z: c, label: `(${a.toFixed(2)},${b.toFixed(2)},${c.toFixed(2)})` });
    }
  }

  return voxels;
}

/* appendBubble is defined above — this duplicate removed */

function scrollChat() {
  const m = ge('chat-msgs');
  m.scrollTop = m.scrollHeight;
}

/* ══════════════════════════════════════════════════════
   CLOCK & POLLING INTERVALS
══════════════════════════════════════════════════════ */
function updateClock() {
  ge('sb-time').textContent = new Date().toTimeString().slice(0, 8);
}
setInterval(updateClock, 1000);
updateClock();
setInterval(refreshNeo4j, 15000);
setInterval(fetchAlerts,  20000);

/* ══════════════════════════════════════════════════════
   BOOT
══════════════════════════════════════════════════════ */
window.addEventListener('DOMContentLoaded', () => {
  _initMarked();
  initIcons();
  initChart();
  initThree();
  initWires();
  initNodeDrag();
  initFloatPanelDrag();
  connectWS();
  updateChatContext();

  /* Neo4j dot: do not wait for assignSensors — probe DB as soon as API is up */
  void refreshNeo4j();

  // Fit all nodes into view after layout has settled
  setTimeout(() => nfZoomFit(), 200);

  // Hydrate from API
  setTimeout(async () => {
    // Always apply the current HTML input values on boot.
    // The input defaults are derived from the physical dome image (exact voxel IDs).
    // This ensures stale Neo4j positions never silently override the correct locations.
    try {
      log('Applying sensor positions from physical dome mapping…', 'info');
      await assignSensors();
    } catch (e) {
      // Server not ready yet — try to at least restore UI state from server
      try {
        const assigned = await apiFetch('/api/sensors/assigned');
        if (assigned.sensors_assigned) {
          onSensorsAssigned({ positions: assigned.positions });
        }
      } catch (_) { /* offline */ }
    }

    await refreshNeo4j();
    await fetchAlerts();

    /* Probe FEM model and chat agent status explicitly */
    try {
      const femSt = await apiFetch('/api/fem/status');
      applyFemReady({ ready: femSt.ready });
    } catch (e) { /* server not ready */ }
    try {
      const agentSt = await apiFetch('/api/chat/status');
      applyAgentReady({ ready: agentSt.ready, error: agentSt.error });
    } catch (e) { /* server not ready */ }

    try {
      const h = await apiFetch('/api/neo4j/sensors/stream');
      if (h.timestamps?.length) updateChart(h);
    } catch (e) { /* Neo4j may not be ready yet */ }

    /* Restore chat history from server (non-blocking, silent fail) */
    await loadChatHistory();
  }, 600);

  /* Update context strip every 5 s in case readings arrive outside WS events */
  setInterval(updateChatContext, 5000);

  log('S2VMO Digital Twin UI initialized — MQTT · Neo4j · FEM', 'ok');
});
