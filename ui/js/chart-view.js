'use strict';
/* ══════════════════════════════════════════════════════
   chart-view.js — MPU Stream + SG Stream charts
   Features:
     • Sliding time-window (10 – 300 points)
     • Drag-to-pan: drag left/right on chart to scrub history
     • Live min/max tracking (actual achieved values)
     • Auto Y-axis scale from the visible window's range
     • LIVE badge dims to amber when panned; dbl-click to snap back
══════════════════════════════════════════════════════ */

/* ── Shared constants ────────────────────────────────── */
const DEFAULT_WINDOW = 60;
const HISTORY_MAX    = 600;

/* ── Chart theme ─────────────────────────────────────── */
const GRID_COLOR = 'rgba(0,0,0,.06)';
const TICK_STYLE = { family: "'Roboto Mono',monospace", size: 8 };

function ds(label, color, dash = []) {
  return {
    label,
    data: [],
    borderColor: color,
    backgroundColor: color + '18',
    borderWidth: 1.75,
    pointRadius: 0,
    tension: 0.35,
    fill: false,
    borderDash: dash,
  };
}

/* ════════════════════════════════════════════════════
   MPU STREAM  — acc_g magnitude + gyro_dps magnitude
════════════════════════════════════════════════════ */
let mpuChart     = null;
let mpuWindow    = DEFAULT_WINDOW;   // visible points (slider)
let mpuPanOffset = 0;                // 0 = live; positive = points into past
let mpuHistory   = { labels: [], datasets: [[], [], [], []] };
let mpuMin       = Infinity;
let mpuMax       = -Infinity;

function initMpuChart() {
  const ctx = ge('mpu-chart').getContext('2d');
  mpuChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        ds('Acc₁ mag (g)',  '#00b191'),
        ds('Acc₂ mag (g)',  '#009070', [4, 2]),
        ds('Gyro₁ (dps)',  '#6c8ebf'),
        ds('Gyro₂ (dps)',  '#4a6fa5', [4, 2]),
      ],
    },
    options: _chartOptions('g / dps'),
  });
  _initChartDrag(
    'mpu-chart',
    () => ({ history: mpuHistory, window: mpuWindow, offset: mpuPanOffset }),
    (newOff) => { mpuPanOffset = newOff; _renderChart(mpuChart, mpuHistory, mpuWindow, mpuPanOffset); _updateLiveBadge('mpu-live', mpuPanOffset); }
  );
}

function pushMpuData(ts, rich) {
  if (!mpuChart || !rich) return;
  const r1 = rich.S1 || {};
  const r2 = rich.S2 || {};

  const accMag  = r => +Math.sqrt((r.acc_g?.x  || 0) ** 2 + (r.acc_g?.y  || 0) ** 2 + (r.acc_g?.z  || 0) ** 2).toFixed(4);
  const gyroMag = r => +Math.sqrt((r.gyro_dps?.x || 0) ** 2 + (r.gyro_dps?.y || 0) ** 2 + (r.gyro_dps?.z || 0) ** 2).toFixed(4);

  const vals = [accMag(r1), accMag(r2), gyroMag(r1), gyroMag(r2)];
  _pushToHistory(mpuHistory, ts.slice(11, 19), vals);

  vals.forEach(v => { if (v < mpuMin) mpuMin = v; if (v > mpuMax) mpuMax = v; });
  _updateMinMaxBadge('mpu-min-lbl', 'mpu-max-lbl', mpuMin, mpuMax, 4);

  /* Only follow live if not panning */
  if (mpuPanOffset === 0) {
    _renderChart(mpuChart, mpuHistory, mpuWindow, 0);
  }
  ge('mpu-ts').textContent = ts.slice(11, 19);
}

function onMpuWindowChange(val) {
  mpuWindow = parseInt(val);
  ge('mpu-window-val').textContent = val + ' pts';
  if (mpuChart) _renderChart(mpuChart, mpuHistory, mpuWindow, mpuPanOffset);
}

/* ════════════════════════════════════════════════════
   SG STREAM  — strain_uE
════════════════════════════════════════════════════ */
let sgChart     = null;
let sgWindow    = DEFAULT_WINDOW;
let sgPanOffset = 0;
let sgHistory   = { labels: [], datasets: [[], []] };
let sgMin       = Infinity;
let sgMax       = -Infinity;

function initSgChart() {
  const ctx = ge('sg-chart').getContext('2d');
  sgChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        ds('SG1 strain (μE)', '#FF8C00'),
        ds('SG2 strain (μE)', '#cc7000', [4, 2]),
      ],
    },
    options: _chartOptions('μE (microstrain)'),
  });
  _initChartDrag(
    'sg-chart',
    () => ({ history: sgHistory, window: sgWindow, offset: sgPanOffset }),
    (newOff) => { sgPanOffset = newOff; _renderChart(sgChart, sgHistory, sgWindow, sgPanOffset); _updateLiveBadge('sg-live', sgPanOffset); }
  );
}

function pushSgData(ts, rich, readings) {
  if (!sgChart) return;
  let s3 = null, s4 = null;
  if (rich) {
    s3 = (rich.S3 || rich.S1 || {}).strain_uE ?? null;
    s4 = (rich.S4 || rich.S2 || {}).strain_uE ?? null;
  } else if (readings) {
    s3 = readings[2] ?? null;
    s4 = readings[3] ?? null;
  }
  if (s3 === null && s4 === null) return;

  const vals = [s3 ?? 0, s4 ?? 0];
  _pushToHistory(sgHistory, ts.slice(11, 19), vals);

  vals.forEach(v => { if (v < sgMin) sgMin = v; if (v > sgMax) sgMax = v; });
  _updateMinMaxBadge('sg-min-lbl', 'sg-max-lbl', sgMin, sgMax, 1);

  if (sgPanOffset === 0) {
    _renderChart(sgChart, sgHistory, sgWindow, 0);
  }
  ge('sg-ts').textContent = ts.slice(11, 19);
}

function onSgWindowChange(val) {
  sgWindow = parseInt(val);
  ge('sg-window-val').textContent = val + ' pts';
  if (sgChart) _renderChart(sgChart, sgHistory, sgWindow, sgPanOffset);
}

/* ════════════════════════════════════════════════════
   CHART DRAG / PAN
   Drag left  → go back in time (offset ↑)
   Drag right → approach live   (offset ↓)
   Double-click → snap back to LIVE
════════════════════════════════════════════════════ */
function _initChartDrag(canvasId, getState, setOffset) {
  const canvas = ge(canvasId);
  if (!canvas) return;

  let dragging    = false;
  let dragStartX  = 0;
  let offsetAtDragStart = 0;

  canvas.style.cursor = 'grab';

  canvas.addEventListener('mousedown', e => {
    e.preventDefault();
    dragging = true;
    dragStartX = e.clientX;
    offsetAtDragStart = getState().offset;
    canvas.style.cursor = 'grabbing';
  });

  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const state = getState();
    const totalPts = state.history.labels.length;
    const winPts   = state.window;

    /* Pixels per data-point: chart area ≈ canvas width − 60px for y-axis */
    const chartWidth = canvas.offsetWidth - 60;
    const pxPerPt = Math.max(1, chartWidth / Math.max(winPts, 1));

    /* Dragging left (negative deltaX) = looking further into the past */
    const deltaX  = e.clientX - dragStartX;
    const deltaPts = Math.round(-deltaX / pxPerPt);
    const maxOff  = Math.max(0, totalPts - winPts);
    const newOff  = Math.max(0, Math.min(maxOff, offsetAtDragStart + deltaPts));
    setOffset(newOff);
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    canvas.style.cursor = getState().offset === 0 ? 'grab' : 'ew-resize';
  });

  /* Double-click snaps back to live */
  canvas.addEventListener('dblclick', () => {
    setOffset(0);
    canvas.style.cursor = 'grab';
  });
}

/* ════════════════════════════════════════════════════
   SHARED HELPERS
════════════════════════════════════════════════════ */
function initChart() {
  initMpuChart();
  initSgChart();
  initFemCycleChart();
}

function _chartOptions(yLabel) {
  return {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        labels: { color: '#555', font: { family: "'Roboto',sans-serif", size: 9 }, boxWidth: 12 },
      },
      tooltip: {
        callbacks: { title: items => items[0]?.label || '' },
      },
    },
    scales: {
      x: {
        ticks: { color: '#aaa', maxTicksLimit: 6, maxRotation: 0, font: TICK_STYLE },
        grid:  { color: GRID_COLOR },
      },
      y: {
        ticks: { color: '#555', font: TICK_STYLE },
        grid:  { color: GRID_COLOR },
        title: { display: true, text: yLabel, color: '#aaa', font: { size: 8 } },
      },
    },
  };
}

function _pushToHistory(hist, label, vals) {
  hist.labels.push(label);
  vals.forEach((v, i) => { if (hist.datasets[i]) hist.datasets[i].push(v ?? 0); });
  /* Cap at HISTORY_MAX */
  if (hist.labels.length > HISTORY_MAX) {
    hist.labels.shift();
    hist.datasets.forEach(d => d.shift());
  }
}

/**
 * Render a slice of history into the chart.
 * @param {number} panOffset  0 = show latest; N = show N points into the past
 */
function _renderChart(chart, hist, window_pts, panOffset = 0) {
  const total = hist.labels.length;
  const n     = Math.min(total, window_pts);

  /* end index: live end = total, panned = total − panOffset */
  const endIdx   = Math.max(n, total - Math.max(0, panOffset));
  const startIdx = Math.max(0, endIdx - n);

  chart.data.labels = hist.labels.slice(startIdx, endIdx);
  chart.data.datasets.forEach((ds, i) => {
    ds.data = (hist.datasets[i] || []).slice(startIdx, endIdx);
  });

  /* Auto-scale Y to visible window */
  const allVals = chart.data.datasets.flatMap(d => d.data).filter(v => isFinite(v));
  if (allVals.length) {
    const visMin = Math.min(...allVals);
    const visMax = Math.max(...allVals);
    const pad    = Math.abs(visMax - visMin) * 0.12 || 0.1;
    chart.options.scales.y.min = visMin - pad;
    chart.options.scales.y.max = visMax + pad;
  }
  chart.update('none');
}

function _updateMinMaxBadge(minId, maxId, min, max, decimals) {
  const fmt = v => isFinite(v) ? v.toFixed(decimals) : '—';
  const minEl = ge(minId), maxEl = ge(maxId);
  if (minEl) minEl.textContent = 'Min ' + fmt(min);
  if (maxEl) maxEl.textContent = 'Max ' + fmt(max);
}

function _updateLiveBadge(badgeId, panOffset) {
  const el = ge(badgeId);
  if (!el) return;
  if (panOffset === 0) {
    el.textContent = 'LIVE';
    el.classList.remove('paused');
  } else {
    el.textContent = `−${panOffset} pts`;
    el.classList.add('paused');
  }
}

/* ════════════════════════════════════════════════════
   FEM CYCLES CHART  — max stress + avg stress per cycle
   X-axis: cycle number  |  Y-axis: stress (Pa)
════════════════════════════════════════════════════ */
let femCycleChart     = null;
let femCycleWindow    = 40;    // cycles shown by default
let femCyclePanOffset = 0;
let femCycleHistory   = { labels: [], datasets: [[], []] };  // [maxStress, avgStress]
let femCycleMin       = Infinity;
let femCycleMax       = -Infinity;

/* Source colour band: track which cycles used MQTT vs defaults */
let femCycleSources   = [];    // 'mqtt' | 'default' per entry, parallel to history

function initFemCycleChart() {
  const ctx = ge('fem-cycle-chart');
  if (!ctx) return;
  femCycleChart = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        ds('Max stress (Pa)',  '#e05252'),           // red
        ds('Avg stress (Pa)',  '#e09020', [5, 3]),   // amber dashed
      ],
    },
    options: _femCycleOptions(),
  });
  _initChartDrag(
    'fem-cycle-chart',
    () => ({ history: femCycleHistory, window: femCycleWindow, offset: femCyclePanOffset }),
    (newOff) => {
      femCyclePanOffset = newOff;
      _renderFemCycleChart();
      _updateLiveBadge('fem-cycle-live', femCyclePanOffset);
    }
  );
}

/** Push one FEM cycle result.
 *  d = { cycle, max_stress_pa, avg_stress_pa, voxels_updated, timestamp, source }
 */
function pushFemCycleData(d) {
  if (!femCycleChart) return;
  const cycle  = d.cycle || 0;
  const maxS   = d.max_stress_pa ?? d.max_stress ?? 0;
  const avgS   = d.avg_stress_pa ?? d.avg_stress ?? 0;
  const label  = `#${cycle}`;
  const source = (d.source || '').toLowerCase().includes('mqtt') ? 'mqtt' : 'default';

  _pushToHistory(femCycleHistory, label, [maxS, avgS]);
  femCycleSources.push(source);
  if (femCycleSources.length > HISTORY_MAX) femCycleSources.shift();

  if (maxS < femCycleMin) femCycleMin = maxS;
  if (maxS > femCycleMax) femCycleMax = maxS;
  _updateMinMaxBadge('fem-cycle-min-lbl', 'fem-cycle-max-lbl', femCycleMin, femCycleMax, 2);

  /* Update voxel badge */
  const voxEl = ge('fem-vox-badge');
  if (voxEl && d.voxels_updated != null) voxEl.textContent = d.voxels_updated.toLocaleString();

  /* Update timestamp badge */
  const tsEl = ge('fem-cycle-ts');
  if (tsEl) tsEl.textContent = `#${cycle}`;

  if (femCyclePanOffset === 0) _renderFemCycleChart();
}

function onFemCycleWindowChange(val) {
  femCycleWindow = parseInt(val);
  const lbl = ge('fem-cycle-window-val');
  if (lbl) lbl.textContent = `${val} cycles`;
  _renderFemCycleChart();
}

function _renderFemCycleChart() {
  if (!femCycleChart) return;
  _renderChart(femCycleChart, femCycleHistory, femCycleWindow, femCyclePanOffset);

  /* Colour the max-stress line red for MQTT cycles, dimmer for default cycles */
  const total    = femCycleHistory.labels.length;
  const n        = Math.min(total, femCycleWindow);
  const endIdx   = Math.max(n, total - Math.max(0, femCyclePanOffset));
  const startIdx = Math.max(0, endIdx - n);
  const srcSlice = femCycleSources.slice(startIdx, endIdx);

  /* Build a per-point colour array for the max-stress dataset */
  const maxDsColors = srcSlice.map(s => s === 'mqtt' ? '#e05252' : '#e0525280');
  femCycleChart.data.datasets[0].borderColor        = maxDsColors;
  femCycleChart.data.datasets[0].backgroundColor    = maxDsColors.map(c => c + '18');
  femCycleChart.data.datasets[0].segment            = {
    borderColor: ctx => srcSlice[ctx.p0DataIndex] === 'mqtt' ? '#e05252' : '#e0525260',
  };
  femCycleChart.update('none');
}

function _femCycleOptions() {
  return {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        labels: { color: '#555', font: { family: "'Roboto',sans-serif", size: 9 }, boxWidth: 12 },
      },
      tooltip: {
        callbacks: {
          title: items => `Cycle ${items[0]?.label || ''}`,
          afterBody: items => {
            const idx = (femCycleHistory.labels.length - femCycleWindow) + (items[0]?.dataIndex || 0);
            const src = femCycleSources[idx] || '—';
            return [`Source: ${src === 'mqtt' ? '📡 MQTT' : '📂 defaults'}`];
          },
        },
      },
    },
    scales: {
      x: {
        ticks: { color: '#aaa', maxTicksLimit: 8, maxRotation: 0, font: TICK_STYLE },
        grid:  { color: GRID_COLOR },
        title: { display: true, text: 'Cycle', color: '#aaa', font: { size: 8 } },
      },
      y: {
        ticks: { color: '#555', font: TICK_STYLE },
        grid:  { color: GRID_COLOR },
        title: { display: true, text: 'Stress (Pa)', color: '#aaa', font: { size: 8 } },
      },
    },
  };
}

/* Legacy compatibility */
function pushChartData(ts, readings) { pushSgData(ts, null, readings); }

function updateChart(history) {
  if (!sgChart || !history.timestamps) return;
  sgHistory.labels   = history.timestamps.map(t => typeof t === 'string' ? t.slice(11, 19) : t);
  sgHistory.datasets = [
    (history.S3 || history.S1 || []).slice(),
    (history.S4 || history.S2 || []).slice(),
  ];
  if (sgHistory.labels.length) {
    const allV = [...sgHistory.datasets[0], ...sgHistory.datasets[1]].filter(isFinite);
    if (allV.length) { sgMin = Math.min(...allV); sgMax = Math.max(...allV); }
    _updateMinMaxBadge('sg-min-lbl', 'sg-max-lbl', sgMin, sgMax, 1);
    _renderChart(sgChart, sgHistory, sgWindow, sgPanOffset);
  }
}
