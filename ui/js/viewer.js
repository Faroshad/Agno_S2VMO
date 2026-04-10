'use strict';
/* ══════════════════════════════════════════════════════
   viewer.js — Three.js r128 voxelised dome viewport
   Material: MeshLambertMaterial — flat ambient-only lighting
             so every voxel face renders identically
══════════════════════════════════════════════════════ */

let renderer, scene, camera, controls;
let instancedMesh     = null;
let instancedEdges    = null;   /* thin edge-stroke overlay */
const dummy = new THREE.Object3D();
let voxelCount = 0, wireframeMode = false;
let _voxelYOffset = 0;
let _origMatrices  = null;

/* ══════════════════════════════════════════════════════
   VOXEL PINS — AI response reference markers
   Each pin = vertical stem Line + circle ring Mesh above the voxel.
   Stored in _pinGroup so all can be shown/hidden with one flag.
══════════════════════════════════════════════════════ */
let _pinGroup   = null;   // THREE.Group — parent of all pin objects
let _pinsVisible = true;  // current toggle state

const PIN_COLOR       = 0xe05a2b;   // vivid orange-red
const PIN_STEM_H      = 6.0;        // stem height in pitch multiples (6 pitches = 0.6 m)
const PIN_RING_R      = 1.8;        // ring radius in pitch multiples (1.8 pitches = 0.18 m)
const PIN_RING_SEGS   = 24;         // smoothness of ring circle

/** Convert a world position into a pin and add it to the scene */
function _buildPin(wx, wy, wz, pitch, label) {
  const g = new THREE.Group();
  const stemH = PIN_STEM_H * pitch;
  const ringR = PIN_RING_R * pitch;

  // Vertical stem (line from voxel top to ring base)
  const stemGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, stemH, 0),
  ]);
  const stemMat = new THREE.LineBasicMaterial({ color: PIN_COLOR, linewidth: 2 });
  g.add(new THREE.Line(stemGeo, stemMat));

  // Flat circle ring at top of stem
  const ringGeo = new THREE.TorusGeometry(ringR, ringR * 0.18, 8, PIN_RING_SEGS);
  const ringMat = new THREE.MeshBasicMaterial({ color: PIN_COLOR, side: THREE.DoubleSide });
  const ring    = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = Math.PI / 2;   // lay flat (horizontal)
  ring.position.y = stemH;
  g.add(ring);

  // Tiny filled disc inside ring so the circle reads clearly
  const discGeo = new THREE.CircleGeometry(ringR * 0.55, PIN_RING_SEGS);
  const discMat = new THREE.MeshBasicMaterial({
    color: PIN_COLOR, transparent: true, opacity: 0.55,
    side: THREE.DoubleSide, depthWrite: false,
  });
  const disc = new THREE.Mesh(discGeo, discMat);
  disc.rotation.x = -Math.PI / 2;
  disc.position.y = stemH;
  g.add(disc);

  g.position.set(wx, wy + (pitch * 0.5), wz);
  g.visible = _pinsVisible;
  g.userData.pinLabel = label || '';
  return g;
}

/**
 * Place pins on voxels extracted from an AI response.
 * @param {Array<{gi?:number, gj?:number, gk?:number, x?:number, y?:number, z?:number, label?:string}>} voxels
 */
function setVoxelPins(voxels) {
  if (!scene) return;
  clearPins();

  if (!voxels || voxels.length === 0) return;
  const pitch = voxelData ? voxelData.pitch : 0.12;

  _pinGroup = new THREE.Group();
  _pinGroup.name = 'voxelPins';

  for (const v of voxels) {
    let wx = null, wy = null, wz = null;

    // Prefer exact world coords if given (from geometry API positions)
    if (v.x != null && v.y != null && v.z != null) {
      wx = v.x;
      wy = v.y + _voxelYOffset;
      wz = v.z;
    }
    // Fall back: grid (i,j,k) + origin + pitch
    else if (v.gi != null && voxelData) {
      const origin = voxelData.origin;
      wx = v.gi * pitch + origin[0];
      wy = v.gj * pitch + origin[1] + _voxelYOffset;
      wz = v.gk * pitch + origin[2];
    }
    // Fall back: look up via _gridKeyToIndex
    else if (v.gi != null && _gridKeyToIndex) {
      const key = `${v.gi},${v.gj},${v.gk}`;
      const idx = _gridKeyToIndex.get(key);
      if (idx !== undefined) {
        wx = _cachedPositions[idx * 3];
        wy = _cachedPositions[idx * 3 + 1] + _voxelYOffset;
        wz = _cachedPositions[idx * 3 + 2];
      }
    }

    if (wx === null) continue;
    _pinGroup.add(_buildPin(wx, wy, wz, pitch, v.label));
  }

  scene.add(_pinGroup);

  // Button state
  const btn = document.getElementById('btn-pins');
  if (btn) {
    btn.classList.toggle('active', _pinsVisible);
    btn.title = `${_pinGroup.children.length} AI voxel pin${_pinGroup.children.length !== 1 ? 's' : ''} (click to toggle)`;
  }
}

/** Remove all current pins from the scene */
function clearPins() {
  if (_pinGroup && scene) {
    scene.remove(_pinGroup);
    _pinGroup.traverse(c => {
      if (c.geometry) c.geometry.dispose();
      if (c.material) {
        if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
        else c.material.dispose();
      }
    });
  }
  _pinGroup = null;
  const btn = document.getElementById('btn-pins');
  if (btn) btn.title = 'AI voxel pins (none yet)';
}

/** Toggle pin visibility */
function togglePins() {
  _pinsVisible = !_pinsVisible;
  if (_pinGroup) _pinGroup.visible = _pinsVisible;
  const btn = document.getElementById('btn-pins');
  if (btn) btn.classList.toggle('active', _pinsVisible);
}

const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();

/* ── Stress: pure greyscale — light grey (low) → dark grey (high) ──
   Slight gamma > 1 pulls mid-tones apart for a smoother perceived gradient. ── */
function stressColor(t) {
  const g = Math.min(Math.max(t, 0), 1);
  const curved = Math.pow(g, 0.88);
  const v = 0.76 - curved * 0.48;
  return new THREE.Color(v, v, v);
}

/* ══════════════════════════════════════════════════════
   STRESS FIELD SMOOTHING (grid neighbours — regional heatmap)
   Raw per-voxel FEM stress is noisy; we diffuse on the solid grid
   (6-connected faces) for several passes, then re-stretch to [0,1]
   so the grey ramp reads as smooth regional zones, not salt-and-pepper.
══════════════════════════════════════════════════════ */
const STRESS_SMOOTH_PASSES     = 5;   /* more passes = softer regions */
const STRESS_RENORMALIZE       = true; /* stretch smoothed field to full grey range */

const NEI6 = [
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
];

/** @param {Float32Array} raw — length voxelCount, API order matches grid indices */
function _smoothStressRegional(raw) {
  const n = raw.length;
  if (!_gridKeyToIndex || !_cachedGridIjk || n !== voxelCount) return new Float32Array(raw);

  let cur = new Float32Array(raw);
  for (let pass = 0; pass < STRESS_SMOOTH_PASSES; pass++) {
    const next = new Float32Array(n);
    for (let vi = 0; vi < n; vi++) {
      const i = _cachedGridIjk[vi * 3];
      const j = _cachedGridIjk[vi * 3 + 1];
      const k = _cachedGridIjk[vi * 3 + 2];
      let sum = cur[vi];
      let cnt = 1;
      for (let e = 0; e < 6; e++) {
        const key = `${i + NEI6[e][0]},${j + NEI6[e][1]},${k + NEI6[e][2]}`;
        if (_gridKeyToIndex.has(key)) {
          sum += cur[_gridKeyToIndex.get(key)];
          cnt++;
        }
      }
      next[vi] = sum / cnt;
    }
    cur = next;
  }

  if (!STRESS_RENORMALIZE) return cur;

  let mn = Infinity, mx = -Infinity;
  for (let vi = 0; vi < n; vi++) {
    const v = cur[vi];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const span = mx - mn;
  if (span < 1e-12) return cur;
  const inv = 1.0 / span;
  const out = new Float32Array(n);
  for (let vi = 0; vi < n; vi++) out[vi] = (cur[vi] - mn) * inv;
  return out;
}

/* ══════════════════════════════════════════════════════
   HEATMAP SPHERE CONFIG
   The sphere is driven by DEVIATION from each sensor's
   own rolling average (not by the absolute reading).
   This makes within-range changes visible:
     quiet/stable  → deviation ≈ 0  → small baseline aura
     shifting load → deviation grows → sphere expands + saturates
     tap / spike   → large deviation → full-saturation flash
══════════════════════════════════════════════════════ */
const HEAT_BASE_R       = 2.5;   // pitches at zero deviation (always-on aura)
const HEAT_MAX_R        = 7.0;   // pitches at full deviation
const HEAT_DEV_MAX      = 35;    // μE deviation from rolling mean → t = 1.0
const HEAT_HISTORY_LEN  = 20;    // samples kept per channel for baseline
const HEAT_BASE_OPACITY = 0.18;  // sphere blend at zero deviation

/* ── Cached decoded geometry buffers (populated once in loadVoxelGeometry) ── */
let _cachedPositions  = null;   // Float32Array  [x,y,z, x,y,z, …] raw world positions (pre Y-offset)
let _cachedSTypeArr   = null;   // Uint8Array    sensor type per voxel: 0=none, 1=MPU, 2=SG
let _cachedIsSensor   = null;   // Uint8Array    1 = sensor voxel
let _cachedStressNorm = null;     // Float32Array — smoothed + renormalized [0..1] for display
let _cachedStressNormRaw = null;  // Float32Array — per-voxel norm from API (tooltip = physical σ)
let _cachedGridIjk = null;        // Int16Array  voxelCount*3  grid indices (i,j,k) per instance
let _gridKeyToIndex = null;       // Map "i,j,k" -> instance index for neighbour lookup

/* ── Sensor sphere centre positions (built after geometry load) ─────────── */
let _sensorWorldPos  = null;           // [{readingIdx, x, y, z, typeCode}, …] — up to 4
let _heatReadings    = [0, 0, 0, 0];   // Current deviation values (μE, per channel)
let _readingHistory  = [[], [], [], []]; // Per-channel ring buffers for rolling baseline

/* ── Heatmap colour palette ──────────────────────────────────────────────── */
const _mpuHot   = new THREE.Color(0x00b191);  // teal   — MPU sphere centre
const _sgHot    = new THREE.Color(0xFF8C00);  // amber  — SG sphere centre
const _baseGray = new THREE.Color(0x909090);  // neutral structural voxel

/* ═══════════════════════════════════════════════════ */
function initThree() {
  const wrap   = ge('three-wrap');
  const canvas = ge('three-canvas');

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputEncoding    = THREE.LinearEncoding;
  renderer.toneMapping       = THREE.NoToneMapping;
  renderer.setClearColor(0xe8e8e8, 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
  resize3D();

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xe8e8e8);

  camera = new THREE.PerspectiveCamera(48, wrap.clientWidth / wrap.clientHeight, 0.1, 200);
  camera.position.set(6, 8, 12);

  /* ── Lighting ──────────────────────────────────────────────────────────── */
  scene.add(new THREE.AmbientLight(0xffffff, 0.92));

  const sun = new THREE.DirectionalLight(0xffffff, 0.17);
  sun.position.set(14, 24, 16);
  sun.castShadow = true;
  sun.shadow.mapSize.width  = 2048;
  sun.shadow.mapSize.height = 2048;
  sun.shadow.camera.near    = 0.5;
  sun.shadow.camera.far     = 50;
  sun.shadow.camera.left    = -14;
  sun.shadow.camera.right   =  14;
  sun.shadow.camera.top     =  14;
  sun.shadow.camera.bottom  = -14;
  sun.shadow.bias           = -0.0002;
  sun.shadow.radius         = 11;
  scene.add(sun);

  /* ── Grid ── */
  const grid = new THREE.GridHelper(40, 40, 0x888888, 0x888888);
  grid.material.opacity    = 0.45;
  grid.material.transparent = true;
  scene.add(grid);

  /* ── Controls ── */
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.minDistance   = 1.5;
  controls.maxDistance   = 80;

  window.addEventListener('resize', resize3D);
  canvas.addEventListener('mousemove', onMouseMove);

  animate();
  loadVoxelGeometry();
}

function resize3D() {
  const w = ge('three-wrap');
  if (!w || !renderer) return;
  renderer.setSize(w.clientWidth, w.clientHeight);
  if (camera) { camera.aspect = w.clientWidth / w.clientHeight; camera.updateProjectionMatrix(); }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

/* ══════════════════════════════════════════════════════
   SENSOR WORLD POSITIONS
   Converts grid-space sensor coords from the API into the
   same Three.js world space used by the InstancedMesh.
══════════════════════════════════════════════════════ */
function _buildSensorPositions(geo) {
  if (!geo.sensor_info || !geo.origin || geo.pitch == null) return;
  const origin = geo.origin;  // [ox, oy, oz]
  const pitch  = geo.pitch;

  _sensorWorldPos = ['S1', 'S2', 'S3', 'S4'].map((id, idx) => {
    const g = geo.sensor_info[id];
    if (!g) return null;
    return {
      readingIdx: idx,
      // Mirror the same transform used in loadVoxelGeometry:
      // three_pos = grid * pitch + origin, then shift Y by _voxelYOffset
      x: g[0] * pitch + origin[0],
      y: g[1] * pitch + origin[1] + _voxelYOffset,
      z: g[2] * pitch + origin[2],
      typeCode: (id === 'S3' || id === 'S4') ? 2 : 1,  // 2=SG, 1=MPU
    };
  }).filter(Boolean);
}

/* ══════════════════════════════════════════════════════
   UNIFIED COLOUR APPLICATION
   Single pass that combines:
     1. Greyscale FEM stress for structural voxels (base)
     2. Heatmap sphere overlay driven by live sensor data
     3. Sensor voxel solid colour with tension-based saturation
   Called after any data update (geometry, stress, MQTT).
══════════════════════════════════════════════════════ */
function _applyAllColors() {
  if (!instancedMesh || !_cachedPositions || !_cachedSTypeArr) return;

  const pitch = voxelData ? voxelData.pitch : 1.0;

  /* _heatReadings now holds signed deviation from rolling baseline per channel.
     Normalise magnitude to [0, 1] against HEAT_DEV_MAX. */
  const tArr = _heatReadings.map(d => Math.min(Math.abs(d) / HEAT_DEV_MAX, 1.0));

  /* Precompute per-sensor sphere radii (in world units) */
  const sensorData = _sensorWorldPos || [];
  const radii = sensorData.map(sp => {
    const t = tArr[sp.readingIdx];
    return (HEAT_BASE_R + t * (HEAT_MAX_R - HEAT_BASE_R)) * pitch;
  });

  const tmpColor = new THREE.Color();

  for (let i = 0; i < voxelCount; i++) {
    const tc = _cachedSTypeArr[i];

    /* ── Sensor voxels: always show sensor colour; saturate with tension ── */
    if (tc > 0) {
      const t = tc === 1
        ? Math.max(tArr[0], tArr[1])   // MPU: max of S1, S2
        : Math.max(tArr[2], tArr[3]);  // SG:  max of S3, S4
      const hot = tc === 1 ? _mpuHot : _sgHot;
      /* 0.35 at idle (clearly visible but not glaring) → 1.0 at full tension */
      instancedMesh.setColorAt(i, tmpColor.lerpColors(_baseGray, hot, 0.35 + t * 0.65));
      continue;
    }

    /* ── Structural voxels ─────────────────────────────────────────────── */
    /* Base: FEM stress greyscale (if available), otherwise neutral grey */
    const base = _cachedStressNorm
      ? stressColor(_cachedStressNorm[i] || 0)
      : _baseGray.clone();

    /* Skip heatmap computation if no sensors have been located yet */
    if (sensorData.length === 0) {
      instancedMesh.setColorAt(i, base);
      continue;
    }

    /* World position of this voxel (matching InstancedMesh placement) */
    const px = _cachedPositions[i * 3];
    const py = _cachedPositions[i * 3 + 1] + _voxelYOffset;
    const pz = _cachedPositions[i * 3 + 2];

    /* Find the sensor whose sphere contributes most to this voxel.
       Sphere is ALWAYS rendered (even at t≈0) — the baseline aura
       stays small and desaturated, growing vivid and larger with tension. */
    let maxW     = 0;
    let hotColor = _baseGray;

    for (let si = 0; si < sensorData.length; si++) {
      const sp     = sensorData[si];
      const t      = tArr[sp.readingIdx];
      const radius = radii[si];           // already accounts for t via precomputed array

      const dx = px - sp.x;
      const dy = py - sp.y;
      const dz = pz - sp.z;
      const d2 = dx*dx + dy*dy + dz*dz;
      if (d2 >= radius * radius) continue;  // outside sphere — skip

      /* Smoothstep gradient: 1 at sphere centre → 0 at sphere edge */
      const f = 1.0 - Math.sqrt(d2) / radius;
      const smooth = f * f * (3.0 - 2.0 * f);

      /* Blend weight:
           - at t=0 → HEAT_BASE_OPACITY  (subtle always-on aura)
           - at t=1 → 1.0               (fully saturated vivid sphere)
         Multiplied by smooth so the gradient fades outward from centre. */
      const w = smooth * (HEAT_BASE_OPACITY + t * (1.0 - HEAT_BASE_OPACITY));

      if (w > maxW) {
        maxW     = w;
        hotColor = sp.typeCode === 2 ? _sgHot : _mpuHot;
      }
    }

    if (maxW > 0.008) {
      /* Blend FEM-stress base colour toward the sensor hot colour */
      instancedMesh.setColorAt(i, base.clone().lerp(hotColor, maxW));
    } else {
      instancedMesh.setColorAt(i, base);
    }
  }

  instancedMesh.instanceColor.needsUpdate = true;
}

/* ══════════════════════════════════════════════════════
   PUBLIC API — called from app.js on each live data event

   Computes per-channel deviation from a rolling mean so
   the sphere reacts to *change*, not just absolute value.
   Channel baseline drifts with the sensor naturally,
   so a tap or load shift always pops out visually.
══════════════════════════════════════════════════════ */
function updateSensorHeatmap(readings) {
  if (!instancedMesh || !voxelData) return;
  if (!readings || readings.length < 4) return;

  for (let i = 0; i < 4; i++) {
    const hist = _readingHistory[i];
    hist.push(readings[i]);
    if (hist.length > HEAT_HISTORY_LEN) hist.shift();

    if (hist.length < 4) {
      /* Not enough history yet — show absolute value so sphere appears immediately */
      _heatReadings[i] = readings[i];
    } else {
      /* Rolling mean over last HEAT_HISTORY_LEN samples */
      const mean = hist.reduce((s, v) => s + v, 0) / hist.length;
      /* Signed deviation: positive = tension spike, negative = compression spike */
      _heatReadings[i] = readings[i] - mean;
    }
  }

  _applyAllColors();
}

/* Reset history on MQTT reconnect so the old baseline doesn't pollute the new session */
function resetHeatmapHistory() {
  _readingHistory = [[], [], [], []];
  _heatReadings   = [0, 0, 0, 0];
}

/* ── Build InstancedMesh from Neo4j geometry ────────────────────────── */
async function loadVoxelGeometry() {
  try {
    const geo = await apiFetch('/api/voxels/geometry');
    if (!geo || !geo.positions_f32_b64) {
      log('Voxel geometry not available — run initialize_neo4j.py first', 'warn');
      return;
    }

    voxelData  = geo;
    voxelCount = geo.count;

    /* Decode and cache buffers (done once, reused by _applyAllColors) */
    _cachedPositions = decodeF32(geo.positions_f32_b64);
    _cachedIsSensor  = decodeU8(geo.is_sensor_u8_b64);
    _cachedSTypeArr  = geo.sensor_type_u8_b64
      ? decodeU8(geo.sensor_type_u8_b64)
      : new Uint8Array(voxelCount);

    /* Grid (i,j,k) per instance — used to smooth stress across solid neighbours */
    _cachedGridIjk = null;
    _gridKeyToIndex = null;
    if (geo.indices_i16_b64 && typeof decodeI16 === 'function') {
      const ij = decodeI16(geo.indices_i16_b64);
      if (ij.length === voxelCount * 3) {
        _cachedGridIjk = ij;
        _gridKeyToIndex = new Map();
        for (let vi = 0; vi < voxelCount; vi++) {
          const gi = ij[vi * 3], gj = ij[vi * 3 + 1], gk = ij[vi * 3 + 2];
          _gridKeyToIndex.set(`${gi},${gj},${gk}`, vi);
        }
      }
    }

    const pitch = geo.pitch;

    /* Bottom face on grid (y = 0) */
    let minY = Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (_cachedPositions[i * 3 + 1] < minY) minY = _cachedPositions[i * 3 + 1];
    }
    _voxelYOffset = -(minY - pitch * 0.5);

    /* Build sensor world-space positions AFTER _voxelYOffset is known */
    _buildSensorPositions(geo);

    const geom = new THREE.BoxGeometry(pitch, pitch, pitch);
    const mat  = new THREE.MeshLambertMaterial({ color: 0xffffff, vertexColors: false });

    instancedMesh = new THREE.InstancedMesh(geom, mat, voxelCount);
    instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    instancedMesh.castShadow    = true;
    instancedMesh.receiveShadow = true;

    /* ── Edge stroke overlay ─────────────────────────────────────────────
       One merged LineSegments object for all voxel outlines — cheaper
       than per-instance lines because it's a single draw call.
    ───────────────────────────────────────────────────────────────────── */
    const edgeGeom = new THREE.EdgesGeometry(geom);
    const edgeMat  = new THREE.LineBasicMaterial({
      color: 0x333333, transparent: true, opacity: 0.55,
    });
    const edgePositions = [];
    const edgePos       = edgeGeom.attributes.position;
    const tempMatrix    = new THREE.Matrix4();
    const tempVec       = new THREE.Vector3();

    _origMatrices = new Float32Array(voxelCount * 16);

    for (let i = 0; i < voxelCount; i++) {
      dummy.position.set(
        _cachedPositions[i * 3],
        _cachedPositions[i * 3 + 1] + _voxelYOffset,
        _cachedPositions[i * 3 + 2]
      );
      dummy.updateMatrix();
      instancedMesh.setMatrixAt(i, dummy.matrix);
      dummy.matrix.toArray(_origMatrices, i * 16);

      /* Initial colour: sensor type colour or neutral grey */
      const tc = _cachedSTypeArr[i];
      instancedMesh.setColorAt(i,
        tc === 2 ? _sgHot : tc === 1 ? _mpuHot : _baseGray
      );

      /* Accumulate edge vertices into merged buffer */
      tempMatrix.copy(dummy.matrix);
      for (let v = 0; v < edgePos.count; v++) {
        tempVec.fromBufferAttribute(edgePos, v).applyMatrix4(tempMatrix);
        edgePositions.push(tempVec.x, tempVec.y, tempVec.z);
      }
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
    instancedMesh.instanceColor.needsUpdate  = true;
    scene.add(instancedMesh);

    /* Apply initial heatmap colours now that geometry + sensor positions are ready.
       This shows the baseline spheres even before the first MQTT packet arrives. */
    _applyAllColors();

    /* Merged edge geometry */
    const mergedEdgeGeom = new THREE.BufferGeometry();
    mergedEdgeGeom.setAttribute('position',
      new THREE.Float32BufferAttribute(edgePositions, 3));
    instancedEdges = new THREE.LineSegments(mergedEdgeGeom, edgeMat);
    instancedEdges.renderOrder = 1;
    scene.add(instancedEdges);

    /* Frame model in camera */
    const box    = new THREE.Box3().setFromObject(instancedMesh);
    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    controls.target.copy(center);
    camera.position.set(
      center.x + maxDim * 0.55,
      center.y + maxDim * 0.45,
      center.z + maxDim * 0.9
    );
    controls.update();

    log(`Loaded ${voxelCount.toLocaleString()} voxels — waiting for live MQTT data`, 'ok');
    ge('db-solid').textContent = `Solid: ${voxelCount.toLocaleString()}`;

  } catch (e) {
    log(`Voxel load error: ${e.message}`, 'err');
  }
}

/* ── Apply FEM stress colours (called on each sim cycle) ────────────── */
async function loadStressData() {
  if (!instancedMesh) return;
  try {
    const s = await apiFetch('/api/voxels/stress');
    if (!s.available || !s.stress_norm_f32_b64) return;
    stressData = s;

    const raw = decodeF32(s.stress_norm_f32_b64);
    _cachedStressNormRaw = raw;
    /* Regional smoothing on grid (neighbour average) → smooth grey heatmap, not speckled */
    _cachedStressNorm = _smoothStressRegional(raw);

    /* Re-render all colours combining stress + current heatmap readings */
    _applyAllColors();

    log(`Stress colouring applied (max ${fmtPa(s.max_pa)})`, 'ok');
  } catch (e) { /* stress not ready yet */ }
}

async function refreshVoxels() { await loadStressData(); }

function resetCamera() {
  if (!instancedMesh) return;
  const box    = new THREE.Box3().setFromObject(instancedMesh);
  const center = box.getCenter(new THREE.Vector3());
  const size   = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  controls.target.copy(center);
  camera.position.set(
    center.x + maxDim * 0.55,
    center.y + maxDim * 0.45,
    center.z + maxDim * 0.9
  );
  controls.update();
}

function toggleWireframe() {
  wireframeMode = !wireframeMode;
  if (instancedMesh) instancedMesh.material.wireframe = wireframeMode;
}

function updateVoxelVisibility() {
  if (!instancedMesh || !stressData || !voxelData || !_origMatrices) return;
  let norm = _cachedStressNorm;
  if (!norm && stressData.stress_norm_f32_b64) {
    const r = decodeF32(stressData.stress_norm_f32_b64);
    norm = _smoothStressRegional(r);
  }
  if (!norm) return;
  const showMPU   = ge('show-sensors').checked;
  const showSG    = ge('show-sg') ? ge('show-sg').checked : true;
  const showH     = ge('show-high').checked;
  const showL     = ge('show-low').checked;
  const hide      = new THREE.Matrix4().makeTranslation(0, -9999, 0);

  for (let i = 0; i < Math.min(norm.length, voxelCount); i++) {
    const t  = norm[i];
    const tc = _cachedSTypeArr ? _cachedSTypeArr[i] : 0;
    let vis = true;
    if (tc === 1 && !showMPU)                      vis = false;
    if (tc === 2 && !showSG)                        vis = false;
    if (tc === 0 && t >  stressThreshold && !showH) vis = false;
    if (tc === 0 && t <= stressThreshold && !showL) vis = false;

    if (vis) {
      dummy.matrix.fromArray(_origMatrices, i * 16);
      instancedMesh.setMatrixAt(i, dummy.matrix);
    } else {
      instancedMesh.setMatrixAt(i, hide);
    }
  }
  instancedMesh.instanceMatrix.needsUpdate = true;
}

function onThreshChange(val) {
  stressThreshold = parseInt(val) / 100;
  ge('thresh-val').textContent = val + '%';
}

function onMouseMove(e) {
  if (!instancedMesh || !voxelData) return;
  const wrap = ge('three-wrap'), rect = wrap.getBoundingClientRect();
  mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(instancedMesh);
  const tip  = ge('voxel-tooltip');
  if (hits.length) {
    const idx     = hits[0].instanceId;
    const tc      = _cachedSTypeArr ? _cachedSTypeArr[idx] : 0;
    const typeTag = tc === 1 ? ' [MPU]' : tc === 2 ? ' [SG]' : '';
    const px = _cachedPositions ? _cachedPositions[idx*3]   : 0;
    const py = _cachedPositions ? _cachedPositions[idx*3+1] : 0;
    const pz = _cachedPositions ? _cachedPositions[idx*3+2] : 0;
    let txt = `Voxel #${idx}${typeTag}  (${px.toFixed(2)}, ${py.toFixed(2)}, ${pz.toFixed(2)})`;
    if (stressData && (_cachedStressNormRaw || _cachedStressNorm)) {
      const nRaw = _cachedStressNormRaw ? _cachedStressNormRaw[idx] : _cachedStressNorm[idx];
      const pa = stressData.min_pa + nRaw * (stressData.max_pa - stressData.min_pa);
      txt += `  σ: ${fmtPa(pa)}`;
    }
    /* Show live tension for sensor voxels */
    if (tc > 0) {
      const t = tc === 1
        ? Math.max(_heatReadings[0], _heatReadings[1])
        : Math.max(_heatReadings[2], _heatReadings[3]);
      txt += `  live: ${t.toFixed(1)} μE`;
    }
    tip.textContent = txt;
    tip.classList.add('show');
  } else {
    tip.classList.remove('show');
  }
}
