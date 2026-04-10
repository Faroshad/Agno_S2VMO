'use strict';
/* ══════════════════════════════════════════════════════
   utils.js — Shared helpers & DOM utilities
   Loaded first; all other modules depend on these.
══════════════════════════════════════════════════════ */

/** Shorthand for document.getElementById */
const ge = id => document.getElementById(id);

/** Set a neumorphic dot state: '', 'on', 'warn' — preserves other classes (e.g. node-status) */
function setDot(id, state) {
  const d = ge(id);
  if (!d) return;
  d.classList.remove('on', 'warn');
  if (state === 'on') d.classList.add('on');
  else if (state === 'warn') d.classList.add('warn');
}

/** Set text content of an element by id */
function setStatus(id, text) {
  const el = ge(id);
  if (el) el.textContent = text;
}

/** Format a number with k/M suffix */
function fmt(n) {
  if (n == null || isNaN(n)) return '—';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return n.toFixed(1);
}

/** Format a Pascal value with Pa/kPa/MPa suffix */
function fmtPa(pa) {
  if (!pa && pa !== 0) return '—';
  if (Math.abs(pa) >= 1e6) return (pa / 1e6).toFixed(2) + 'MPa';
  if (Math.abs(pa) >= 1e3) return (pa / 1e3).toFixed(1) + 'kPa';
  return pa.toFixed(1) + 'Pa';
}

/** Decode base64 → Float32Array */
function decodeF32(b64) {
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Float32Array(buf.buffer);
}

/** Decode base64 → Uint8Array */
function decodeU8(b64) {
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf;
}

/** Decode base64 → Int16Array (grid indices i,j,k triplets) */
function decodeI16(b64) {
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Int16Array(buf.buffer);
}

/** JSON fetch wrapper — throws on HTTP error */
async function apiFetch(path, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  if (!r.ok) { const t = await r.text(); throw new Error(t || r.statusText); }
  return r.json();
}

/** Append a log entry to the pipeline log node */
function log(text, level = 'info') {
  const panel = ge('log-panel');
  if (!panel) return;
  const div = document.createElement('div');
  div.className = `log-entry ${level}`;
  div.innerHTML = `<span class="lt">${new Date().toTimeString().slice(0, 8)}</span>${text}`;
  panel.insertBefore(div, panel.firstChild);
  while (panel.children.length > 60) panel.removeChild(panel.lastChild);
}

/** Show the modal dialog */
function showModal(title, html) {
  ge('modal-title').textContent = title;
  ge('modal-body').innerHTML = html;
  ge('modal-overlay').classList.add('show');
}

/** Close the modal dialog */
function closeModal() {
  ge('modal-overlay').classList.remove('show');
}

/** Inline SVG alert icon (critical or warning) */
function alertIconSvg(sev) {
  return sev === 'critical'
    ? `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`
    : `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
}
