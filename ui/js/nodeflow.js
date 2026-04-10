'use strict';
/* ══════════════════════════════════════════════════════
   nodeflow.js — Zoomable/pannable node canvas,
                 wire routing, node drag, float-panel drag
   Depends on: utils.js (ge)
══════════════════════════════════════════════════════ */

/* ── Zoom / Pan state ─────────────────────────────── */
let nfScale = 1, nfTx = 0, nfTy = 0;
let _panStart = null, _panDragging = false;

/** Apply current transform to #nf-canvas and redraw wires */
function applyNFTransform() {
  ge('nf-canvas').style.transform = `translate(${nfTx}px,${nfTy}px) scale(${nfScale})`;
  ge('nf-zoom-label').textContent = Math.round(nfScale * 100) + '%';
  updateAllWires();
}

/** Zoom by a factor, keeping the viewport centre fixed */
function nfZoomStep(factor) {
  const nf = ge('nodeflow').getBoundingClientRect();
  const cx = nf.width / 2, cy = nf.height / 2;
  nfTx = cx - (cx - nfTx) * factor;
  nfTy = cy - (cy - nfTy) * factor;
  nfScale = Math.max(0.15, Math.min(4, nfScale * factor));
  applyNFTransform();
}

/** Fit all nodes into the visible viewport */
function nfZoomFit() {
  const nf = ge('nodeflow');
  const nodes = nf.querySelectorAll('.node');
  if (!nodes.length) return;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  nodes.forEach(n => {
    const l = parseInt(n.style.left) || 0;
    const t = parseInt(n.style.top)  || 0;
    const w = n.offsetWidth  || 200;
    const h = n.offsetHeight || 180;
    minX = Math.min(minX, l);     minY = Math.min(minY, t);
    maxX = Math.max(maxX, l + w); maxY = Math.max(maxY, t + h);
  });
  const pad = 40;
  const fw = nf.clientWidth  - pad * 2;
  const fh = nf.clientHeight - pad * 2;
  const s  = Math.min(fw / (maxX - minX + 1e-6), fh / (maxY - minY + 1e-6), 2);
  nfScale = Math.max(0.15, s);
  nfTx = pad - minX * nfScale;
  nfTy = pad - minY * nfScale;
  applyNFTransform();
}

/* ── Wheel zoom ───────────────────────────────────── */
ge('nodeflow').addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  const rect = ge('nodeflow').getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  nfTx = cx - (cx - nfTx) * factor;
  nfTy = cy - (cy - nfTy) * factor;
  nfScale = Math.max(0.15, Math.min(4, nfScale * factor));
  applyNFTransform();
}, { passive: false });

/* ── Pan on empty area ────────────────────────────── */
ge('nodeflow').addEventListener('mousedown', e => {
  const tgt = e.target;
  const isEmpty = (
    tgt === ge('nodeflow') ||
    tgt === ge('nf-canvas') ||
    tgt === ge('wire-svg') ||
    tgt.tagName === 'svg'
  );
  if (isEmpty) {
    _panStart = { sx: e.clientX - nfTx, sy: e.clientY - nfTy };
    _panDragging = true;
    ge('nodeflow').style.cursor = 'grabbing';
    e.preventDefault();
  }
});

document.addEventListener('mousemove', e => {
  if (_panDragging && _panStart) {
    nfTx = e.clientX - _panStart.sx;
    nfTy = e.clientY - _panStart.sy;
    applyNFTransform();
  }
});

document.addEventListener('mouseup', () => {
  if (_panDragging) {
    _panDragging = false;
    _panStart = null;
    ge('nodeflow').style.cursor = '';
  }
});

/* ══════════════════════════════════════════════════════
   WIRES
══════════════════════════════════════════════════════ */
let wireMap = {};

const WIRE_DEFS = [
  { id: 'wire-mqtt-sens',   fport: 'p-mqtt-out',  tport: 'p-sens-in1' },
  { id: 'wire-mqtt-mpu',    fport: 'p-mqtt-out',  tport: 'p-mpu-in'   },  /* MQTT → MPU Stream */
  { id: 'wire-sens-fem',    fport: 'p-sens-out',  tport: 'p-fem-in'   },
  { id: 'wire-fem-neo4j',   fport: 'p-fem-out',   tport: 'p-neo4j-in' },
  { id: 'wire-neo4j-sg',    fport: 'p-neo4j-out', tport: 'p-sg-in2'   },  /* Neo4j → SG Stream */
  { id: 'wire-fem-sg',      fport: 'p-fem-out2',  tport: 'p-sg-in'    },  /* FEM → SG Stream  */
  { id: 'wire-fem-stats',    fport: 'p-fem-out2',  tport: 'p-stats-in'    },
  { id: 'wire-neo4j-log',    fport: 'p-neo4j-out', tport: 'p-log-in'      },
  { id: 'wire-fem-femcycle', fport: 'p-fem-out3',  tport: 'p-femcycle-in' },
];

/** Build SVG path elements and register in wireMap */
function initWires() {
  const svg = ge('wire-svg');
  WIRE_DEFS.forEach(w => {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('class', 'wire');
    path.setAttribute('id', w.id);
    svg.appendChild(path);
    wireMap[w.id] = { ...w, path };
  });
  updateAllWires();
}

/**
 * Return the centre of a port element in canvas-local (SVG) coordinates.
 * Accounts for the current zoom/pan transform.
 */
function portCenter(id) {
  const p = ge(id);
  if (!p) return { x: 0, y: 0 };
  const r  = p.getBoundingClientRect();
  const nf = ge('nodeflow').getBoundingClientRect();
  const sx = r.left - nf.left + r.width  / 2;
  const sy = r.top  - nf.top  + r.height / 2;
  return { x: (sx - nfTx) / nfScale, y: (sy - nfTy) / nfScale };
}

/** Redraw all wire bezier paths */
function updateAllWires() {
  Object.values(wireMap).forEach(w => {
    const A = portCenter(w.fport), B = portCenter(w.tport);
    const dx = Math.abs(B.x - A.x) * 0.5;
    w.path.setAttribute('d',
      `M${A.x},${A.y} C${A.x + dx},${A.y} ${B.x - dx},${B.y} ${B.x},${B.y}`);
  });
}

/** Activate / deactivate a wire's visual style */
function wireActive(id, active) {
  const w = wireMap[id];
  if (w) w.path.className.baseVal = `wire ${active ? 'active' : ''}`;
}

/* ══════════════════════════════════════════════════════
   NODE DRAG  (inside #nf-canvas)
   Nodes are free to go anywhere in the infinite canvas —
   no clamping, just pure canvas-space translation.
══════════════════════════════════════════════════════ */
function initNodeDrag() {
  document.querySelectorAll('.node').forEach(node => {
    let ox = 0, oy = 0, dragging = false;

    /* Allow dragging from the node-head OR anywhere on the node body
       that isn't an interactive element (button / input / select). */
    const onDown = e => {
      if (e.target.closest('button, input, select, textarea, a')) return;
      e.preventDefault();
      e.stopPropagation();
      dragging = true;
      const r = node.getBoundingClientRect();
      ox = (e.clientX - r.left) / nfScale;
      oy = (e.clientY - r.top)  / nfScale;
      node.style.zIndex = 50;
      node.style.cursor = 'grabbing';
    };

    node.querySelector('.node-head').addEventListener('mousedown', onDown);
    node.addEventListener('mousedown', onDown);

    document.addEventListener('mousemove', e => {
      if (!dragging) return;
      const nf = ge('nodeflow').getBoundingClientRect();
      /* Transform screen coords → canvas coords with no clamping */
      const lx = (e.clientX - nf.left - nfTx) / nfScale - ox;
      const ly = (e.clientY - nf.top  - nfTy) / nfScale - oy;
      node.style.left = lx + 'px';
      node.style.top  = ly + 'px';
      updateAllWires();
    });

    document.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      node.style.zIndex  = '';
      node.style.cursor  = '';
    });
  });
}

/* ══════════════════════════════════════════════════════
   FLOATING PANEL DRAG  (bottom 3D scene section)
   Panels stay mostly within their parent but can freely
   go all the way to the edges; only a 20px strip at the
   bottom keeps the drag handle visible.
══════════════════════════════════════════════════════ */
function initFloatPanelDrag() {
  document.querySelectorAll('.float-panel').forEach(panel => {
    const head = panel.querySelector('.fp-head');
    if (!head) return;
    let ox = 0, oy = 0, dragging = false;

    head.addEventListener('mousedown', e => {
      if (e.target.closest('button, input, select')) return;
      e.preventDefault();
      dragging = true;
      const r = panel.getBoundingClientRect();
      ox = e.clientX - r.left;
      oy = e.clientY - r.top;
      panel.style.zIndex  = 50;
      panel.style.right   = 'auto';
      panel.style.bottom  = 'auto';
      head.style.cursor   = 'grabbing';
    });

    document.addEventListener('mousemove', e => {
      if (!dragging) return;
      const pr  = ge('dome-section').getBoundingClientRect();
      const pw  = panel.offsetWidth;
      const ph  = panel.offsetHeight;
      const nx  = e.clientX - pr.left - ox;
      const ny  = e.clientY - pr.top  - oy;
      /* Clamp so at least 20 px of the panel stays inside on each edge */
      const MARGIN = 20;
      panel.style.left = Math.max(-pw + MARGIN, Math.min(nx, pr.width  - MARGIN)) + 'px';
      panel.style.top  = Math.max(-ph + MARGIN, Math.min(ny, pr.height - MARGIN)) + 'px';
    });

    document.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      panel.style.zIndex = '';
      head.style.cursor  = '';
    });
  });
}
