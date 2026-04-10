#!/usr/bin/env python3
"""
Structural Health Monitoring Pipeline
Converts sensor readings to 3D stress fields using two-stage neural network approach.

Usage:
    python pipeline.py
"""

import numpy as np
import torch
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
from neo4j import GraphDatabase
from agno.agent import Toolkit
# Support both package-relative and absolute imports
try:
    from ..config.settings import Settings
except ImportError:
    from config.settings import Settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_VOXEL_GRID_PATH = PROJECT_ROOT / "out" / "voxel_grid.npz"


# ============================================================
# MODULE-LEVEL SINGLETONS
# Loaded once on first call; kept alive for the process lifetime.
# Reloading models from disk on every cycle is:
#   (a) very slow (hundreds of MB read each time)
#   (b) the root cause of apparent "random" field changes —
#       torch.load() re-initialises buffers and re-maps device
#       tensors, which introduces subtle floating-point variability
#       that the large FC weight matrices amplify into large
#       per-voxel deltas even for identical sensor inputs.
# ============================================================
_loaded_s2v: Optional[Any] = None          # SensorToVoxelNet instance
_loaded_v2s: Optional[Any] = None          # StressUNet instance
_loaded_device: Optional[Any] = None       # torch.device used when models were loaded
_loaded_strain_mean: Optional[np.ndarray] = None
_loaded_strain_std:  Optional[np.ndarray] = None
_loaded_voxel_mask: Optional[np.ndarray] = None   # cached boolean M
_loaded_coords:     Optional[np.ndarray] = None   # cached argwhere(M)

# ── Temporal-smoothing state ─────────────────────────────────
# Stores the stress / strain field from the most recent FEM cycle.
# Used to smooth out NN noise when sensors are not changing much.
_prev_stress_voxel:   Optional[np.ndarray] = None  # (6, X, Y, Z)
_prev_strain_voxel:   Optional[np.ndarray] = None  # (3, X, Y, Z)
_prev_sensor_reading: Optional[np.ndarray] = None  # (4,) in μE

# ── Smoothing hyper-parameters ───────────────────────────────
# These control how aggressively the EMA blends old vs new output.
# If sensor change < GATE_THRESHOLD_uE → reuse previous field entirely.
# Otherwise alpha (blend weight for new field) rises from ALPHA_MIN
# to ALPHA_MAX as sensor delta grows from 0 → ALPHA_FULL_uE.
_GATE_THRESHOLD_uE = 1.5     # μE: max per-sensor change to skip NN entirely
_ALPHA_MIN         = 0.08    # minimum new-field weight (barely changing sensors)
_ALPHA_MAX         = 0.80    # maximum new-field weight (rapidly changing sensors)
_ALPHA_FULL_uE     = 80.0    # μE sensor delta at which alpha reaches _ALPHA_MAX


def _sensor_delta_uE(curr: np.ndarray, prev: np.ndarray) -> float:
    """Return the maximum per-sensor absolute change in μE."""
    return float(np.max(np.abs(curr - prev)))


def _adaptive_alpha(delta_uE: float) -> float:
    """
    Map sensor change (μE) → EMA blend weight for the NEW stress field.

    Curve: power-law ramp from _ALPHA_MIN at delta=0 to _ALPHA_MAX at
    delta >= _ALPHA_FULL_uE.  This keeps the stress field very stable
    when sensors are quiet and allows fast tracking during real events.
    """
    t = min(delta_uE / _ALPHA_FULL_uE, 1.0) ** 0.6   # sub-linear ramp
    return float(np.clip(_ALPHA_MIN + t * (_ALPHA_MAX - _ALPHA_MIN),
                         _ALPHA_MIN, _ALPHA_MAX))


def _load_models_once():
    """
    Load (or return already-loaded) model weights.

    Called at the top of run_pipeline(); no-op if models are already in
    memory.  Using a singleton avoids repeated disk I/O and guarantees
    that the same weight tensors are reused every cycle, making NN
    output deterministic for identical inputs.
    """
    global _loaded_s2v, _loaded_v2s, _loaded_device
    global _loaded_strain_mean, _loaded_strain_std
    global _loaded_voxel_mask, _loaded_coords

    if _loaded_s2v is not None:
        return (_loaded_s2v, _loaded_v2s, _loaded_device,
                _loaded_strain_mean, _loaded_strain_std,
                _loaded_voxel_mask, _loaded_coords)

    print("🔄 [FEM] Loading models into memory (one-time cost)…")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Force deterministic CUDA ops so identical inputs always produce
    # identical outputs (matters if CUDA is available).
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # Load voxel mask once
    vg     = np.load(_require_file(DEFAULT_VOXEL_GRID_PATH, "voxel grid"), allow_pickle=True)
    M      = vg["matrix"].astype(bool)
    coords = np.argwhere(M)

    # Build and load S2V model
    s2v = SensorToVoxelNet(n_sensors=4, grid_shape=(83, 45, 87)).to(device)
    s2v.load_state_dict(
        torch.load(_require_file(MODELS_DIR / "s2v_model_final.pth", "S2V model weights"),
                   map_location=device),
        strict=False,
    )
    s2v.eval()

    # Build and load UNet model
    v2s = StressUNet(in_channels=3, out_channels=6).to(device)
    v2s.load_state_dict(
        torch.load(_require_file(MODELS_DIR / "unet_model_final.pth", "UNet model weights"),
                   map_location=device),
        strict=False,
    )
    v2s.eval()

    # Load normalization stats
    strain_mean = np.load(_require_file(MODELS_DIR / "strain_mean.npy", "strain mean"))
    strain_std  = np.load(_require_file(MODELS_DIR / "strain_std.npy",  "strain std"))

    # Cache globally
    _loaded_s2v         = s2v
    _loaded_v2s         = v2s
    _loaded_device      = device
    _loaded_strain_mean = strain_mean
    _loaded_strain_std  = strain_std
    _loaded_voxel_mask  = M
    _loaded_coords      = coords

    print(f"✅ [FEM] Models loaded on {device} — {int(M.sum())} solid voxels")
    return (s2v, v2s, device, strain_mean, strain_std, M, coords)


def _require_file(file_path: Path, description: str) -> Path:
    """Fail fast with a clear message when required runtime assets are missing."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {file_path}. "
            f"Ensure required assets are present before running FEM analysis."
        )
    return file_path

# ============================================================
# BOOLEAN MASK CREATION
# ============================================================

def create_voxel_mask(voxel_grid_path: Optional[str] = None):
    """
    Create boolean mask M from the voxelized mesh data.
    Uses the same voxelization as the neural network pipeline (0.1m voxels).
    Returns the cached mask if already loaded; reads from disk only once.
    """
    global _loaded_voxel_mask, _loaded_coords
    if _loaded_voxel_mask is not None and voxel_grid_path is None:
        # Return the already-loaded singleton — avoids repeated disk I/O
        return _loaded_voxel_mask

    print("Loading voxel grid from voxelized mesh...")
    if voxel_grid_path is None:
        resolved_voxel_grid_path = _require_file(DEFAULT_VOXEL_GRID_PATH, "voxel grid")
    else:
        resolved_voxel_grid_path = _require_file(Path(voxel_grid_path), "voxel grid")

    vg = np.load(resolved_voxel_grid_path, allow_pickle=True)
    M  = vg["matrix"].astype(bool)
    origin = vg["origin"].astype(float).reshape(3,)
    pitch  = float(np.atleast_1d(vg["pitch"])[0])

    print(f"Voxel matrix shape: {M.shape}")
    print(f"Origin: {origin}")
    print(f"Pitch: {pitch}")
    print(f"Solid voxels: {np.sum(M)}")

    # Cache for subsequent calls
    _loaded_voxel_mask = M
    _loaded_coords     = np.argwhere(M)
    return M

# ============================================================
# MODEL DEFINITIONS
# ============================================================

class SensorToVoxelNet(torch.nn.Module):
    """Sensor readings to 3D strain voxel field network."""
    def __init__(self, n_sensors=4, grid_shape=(83, 45, 87)):
        super().__init__()
        self.n_input = n_sensors
        self.grid_shape = grid_shape
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_sensors, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, np.prod(grid_shape) * 3),
        )

    def forward(self, x):
        B = x.size(0)
        out = self.fc(x)
        return out.view(B, 3, *self.grid_shape)


class StressUNet(torch.nn.Module):
    """3D strain voxel field to 6-component stress field U-Net."""
    def __init__(self, in_channels=3, out_channels=6, base_dim=32):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, base_dim, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(base_dim, base_dim * 2, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv3d(base_dim * 2, base_dim, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(base_dim, out_channels, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ============================================================
# MAIN PIPELINE FUNCTION
# ============================================================

def run_pipeline(sensor_reading=None, output_dir="out"):
    """
    Run the complete sensor-to-stress pipeline.

    Key stability enhancements vs. the naïve version:

    1. **Model singleton** — weights are loaded from disk exactly once and kept
       in memory.  Reloading on every cycle caused subtle floating-point
       variability that the large FC layer amplified into seemingly random
       per-voxel jumps even for identical sensor inputs.

    2. **Sensor-change gate** — if the averaged sensor readings have not changed
       by more than _GATE_THRESHOLD_uE μE since the last cycle, the neural
       network is NOT run; the previous stress / strain field is returned as-is.
       This prevents the model from producing meaningless micro-fluctuations
       when the structure is at rest.

    3. **Adaptive EMA smoothing** — when the sensor delta exceeds the gate
       threshold, the new NN output is blended with the previous field using
       a weight (alpha) that scales with how much the sensors actually changed.
       Tiny changes → alpha ≈ 0.08 (90 % of previous field survives).
       Large changes → alpha ≈ 0.80 (new field dominates).
       This makes the stress visualisation track real events while suppressing
       NN noise during stable periods.

    Args:
        sensor_reading: (1, 4) array of averaged sensor values in microstrain.
                        Must not be None — FEM only runs on real sensor data.
        output_dir: Directory for intermediate .npz outputs.

    Returns:
        dict with strain_voxel, stress_voxel, stress_magnitude, voxel_mask,
        solid_coords, strain/stress output paths, and smoothing metadata.
    """
    global _prev_stress_voxel, _prev_strain_voxel, _prev_sensor_reading

    if sensor_reading is None:
        raise ValueError(
            "sensor_reading is required — pass a (1, 4) array of live strain values "
            "in microstrain (μE). FEM must not run without real sensor data."
        )

    os.makedirs(output_dir, exist_ok=True)

    # ── 0. Load models once (no-op on subsequent calls) ───────────────────────
    s2v, v2s, device, strain_mean, strain_std, M, coords = _load_models_once()

    vg     = np.load(_require_file(DEFAULT_VOXEL_GRID_PATH, "voxel grid"), allow_pickle=True)
    origin = vg["origin"].astype(float).reshape(3,)
    pitch  = float(np.atleast_1d(vg["pitch"])[0])

    # Persist voxel grid to output dir (downstream Neo4j code reads it)
    np.savez(os.path.join(output_dir, "voxel_grid.npz"), matrix=M, origin=origin, pitch=pitch)

    # ── 1. Normalise sensor input (μE → dimensionless SI strain) ──────────────
    sensor_arr = np.array(sensor_reading).reshape(1, -1) * 1e-6   # (1, 4)
    curr_uE    = sensor_arr[0] * 1e6                               # (4,) in μE for comparisons
    print(f"📥 [FEM] Sensor input: {curr_uE} μE")

    # ── 2. Sensor-change gate ─────────────────────────────────────────────────
    smoothing_meta = {"mode": "nn_full", "alpha": 1.0, "delta_uE": 0.0}

    if _prev_sensor_reading is not None:
        delta_uE = _sensor_delta_uE(curr_uE, _prev_sensor_reading)
        smoothing_meta["delta_uE"] = round(delta_uE, 3)
        print(f"📊 [FEM] Sensor delta: {delta_uE:.2f} μE "
              f"(gate threshold: {_GATE_THRESHOLD_uE} μE)")

        if delta_uE < _GATE_THRESHOLD_uE and _prev_stress_voxel is not None:
            # Sensors basically unchanged → reuse previous field directly.
            # No NN inference, no blending — identical output guarantees
            # zero flicker in the 3D viewer when the dome is at rest.
            print(f"⏸  [FEM] Below gate — reusing previous stress field (no NN run)")
            eps_voxel   = _prev_strain_voxel.copy()
            sigma_voxel = _prev_stress_voxel.copy()
            smoothing_meta["mode"] = "gate_reuse"
            smoothing_meta["alpha"] = 0.0

            # Skip straight to output saving
            strain_output_path = os.path.join(output_dir, "eps_voxel_predicted.npz")
            stress_output_path = os.path.join(output_dir, "sigma_voxel_predicted_pp.npz")
            np.savez(strain_output_path, eps_voxel=eps_voxel)
            np.savez(stress_output_path, sigma_voxel=sigma_voxel)

            σ_mag = np.sqrt(np.sum(sigma_voxel[:3]**2, axis=0))
            _prev_sensor_reading = curr_uE.copy()   # still update sensor bookmark
            _log_stress_stats(sigma_voxel)
            return _build_result(eps_voxel, sigma_voxel, σ_mag, M, coords,
                                 strain_output_path, stress_output_path,
                                 output_dir, smoothing_meta)
    else:
        delta_uE = 0.0

    # ── 3. Run NN inference ────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("STAGE 1: Sensor readings → 3D strain voxels")
    print("="*50)
    sensor_tensor = torch.tensor(sensor_arr, dtype=torch.float32).to(device)
    with torch.no_grad():
        eps_voxel = s2v(sensor_tensor).cpu().numpy()[0]  # (3,83,45,87)

    strain_output_path = os.path.join(output_dir, "eps_voxel_predicted.npz")
    np.savez(strain_output_path, eps_voxel=eps_voxel)
    print(f"✅ Stage 1 complete | shape={eps_voxel.shape}")

    print("\n" + "="*50)
    print("STAGE 2: 3D strain voxels → 6-component stress field")
    print("="*50)
    eps_norm = (eps_voxel - strain_mean) / (strain_std + 1e-8)
    X = torch.tensor(eps_norm[None, ...], dtype=torch.float32).to(device)
    with torch.no_grad():
        sigma_voxel = v2s(X).cpu().numpy()[0]  # (6,83,45,87)

    stress_output_path = os.path.join(output_dir, "sigma_voxel_predicted_pp.npz")
    np.savez(stress_output_path, sigma_voxel=sigma_voxel)
    print(f"✅ Stage 2 complete | shape={sigma_voxel.shape}")

    # ── 4. Adaptive EMA smoothing ─────────────────────────────────────────────
    # Only applies when we have a previous field AND delta exceeded the gate.
    if (_prev_stress_voxel is not None
            and _prev_stress_voxel.shape == sigma_voxel.shape
            and _prev_strain_voxel is not None):
        alpha = _adaptive_alpha(delta_uE)
        print(f"🔀 [FEM] EMA blend: alpha={alpha:.3f} "
              f"(delta={delta_uE:.1f} μE → "
              f"{alpha*100:.0f}% new + {(1-alpha)*100:.0f}% previous)")
        sigma_voxel = alpha * sigma_voxel + (1.0 - alpha) * _prev_stress_voxel
        eps_voxel   = alpha * eps_voxel   + (1.0 - alpha) * _prev_strain_voxel
        smoothing_meta["mode"]  = "ema_blend"
        smoothing_meta["alpha"] = round(alpha, 3)

        # Overwrite saved outputs with the smoothed fields
        np.savez(strain_output_path, eps_voxel=eps_voxel)
        np.savez(stress_output_path, sigma_voxel=sigma_voxel)
    else:
        smoothing_meta["mode"] = "nn_first_cycle"

    # ── 5. Cache for next cycle ────────────────────────────────────────────────
    _prev_stress_voxel   = sigma_voxel.copy()
    _prev_strain_voxel   = eps_voxel.copy()
    _prev_sensor_reading = curr_uE.copy()

    # ── 6. Save auxiliary artefacts & log ─────────────────────────────────────
    mask_output_path  = os.path.join(output_dir, "voxel_mask.npz")
    solid_output_path = os.path.join(output_dir, "solid_voxel_data.npz")

    np.savez(mask_output_path,
             M=M, coords=coords,
             solid_voxel_count=len(coords), total_voxels=M.size)

    strain_solid = eps_voxel[:, M]
    stress_solid = sigma_voxel[:, M]
    np.savez(solid_output_path,
             strain_solid=strain_solid, stress_solid=stress_solid,
             coords=coords, M=M)

    σ_mag = np.sqrt(np.sum(sigma_voxel[:3]**2, axis=0))
    _log_stress_stats(sigma_voxel)
    print(f"\n✅ Full pipeline complete! | mode={smoothing_meta['mode']}")

    return _build_result(eps_voxel, sigma_voxel, σ_mag, M, coords,
                         strain_output_path, stress_output_path,
                         output_dir, smoothing_meta)


# ─── Small helpers used by run_pipeline ──────────────────────────────────────

def _log_stress_stats(sigma_voxel: np.ndarray):
    σ_names = ["σ_xx", "σ_yy", "σ_zz", "σ_xy", "σ_yz", "σ_xz"]
    print("\nStress component ranges:")
    for i, name in enumerate(σ_names):
        print(f"  {name:>6}: {sigma_voxel[i].min():>10.3e} ~ {sigma_voxel[i].max():>10.3e}")
    σ_mag = np.sqrt(np.sum(sigma_voxel[:3]**2, axis=0))
    print(f"  Magnitude: mean={σ_mag.mean():.3e}  "
          f"p99={np.percentile(σ_mag, 99):.3e}  max={σ_mag.max():.3e}")


def _build_result(eps_voxel, sigma_voxel, σ_mag, M, coords,
                  strain_path, stress_path, output_dir, meta):
    return {
        "strain_voxel":      eps_voxel,
        "stress_voxel":      sigma_voxel,
        "stress_magnitude":  σ_mag,
        "strain_output_path": strain_path,
        "stress_output_path": stress_path,
        "voxel_mask":        M,
        "solid_coords":      coords,
        "smoothing":         meta,      # mode / alpha / delta_uE for logging
    }


# ============================================================
# NEO4J INTEGRATION FOR FEM RESULTS
# ============================================================

def _get_neo4j_driver():
    """Get Neo4j driver connection"""
    conn_info = Settings.get_connection_info()
    return GraphDatabase.driver(
        conn_info["uri"],
        auth=(conn_info["user"], conn_info["password"])
    )


def update_neo4j_with_fem_results(
    strain_voxel: np.ndarray,
    stress_voxel: np.ndarray,
    voxel_mask: np.ndarray,
    solid_coords: np.ndarray,
    sensor_reading: np.ndarray,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update Neo4j database with FEM analysis results for each voxel.
    
    Stores strain and stress components as time-series data with timestamps.
    Each FEM run creates new entries with version tracking.
    
    Args:
        strain_voxel: (3, X, Y, Z) strain field [ε_xx, ε_yy, ε_zz]
        stress_voxel: (6, X, Y, Z) stress field [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz]
        voxel_mask: Boolean mask indicating solid voxels
        solid_coords: Coordinates of solid voxels
        sensor_reading: Original sensor readings that triggered this analysis
        timestamp: Optional timestamp (default: current time)
        
    Returns:
        Dict with update statistics
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    print(f"\n{'='*50}")
    print("UPDATING NEO4J WITH FEM RESULTS")
    print(f"{'='*50}")
    print(f"Timestamp: {timestamp}")
    
    driver = _get_neo4j_driver()
    conn_info = Settings.get_connection_info()
    
    # Extract strain and stress data for solid voxels only
    strain_solid = strain_voxel[:, voxel_mask]  # (3, N_solid)
    stress_solid = stress_voxel[:, voxel_mask]  # (6, N_solid)
    
    # Calculate stress magnitude
    stress_magnitude = np.sqrt(np.sum(stress_solid[:3]**2, axis=0))
    
    print(f"Processing {len(solid_coords)} solid voxels...")
    
    updated_voxels = []
    
    with driver.session(database=conn_info["database"]) as session:
        print(f"📊 Refreshing FEM results in Neo4j (overwrite mode — no accumulation)")

        # MERGE on fixed id 'latest' so only one FEMAnalysis node ever exists.
        # Increment fem_cycle_count and append timestamp so the chatbot can report
        # how many FEM runs occurred (counting FEMAnalysis nodes always returns 1).
        session.run("""
            MERGE (a:FEMAnalysis {analysis_id: 'latest'})
            SET a.fem_cycle_count = coalesce(a.fem_cycle_count, 0) + 1,
                a.timestamp      = $timestamp,
                a.sensor_reading = $sensor_reading,
                a.total_voxels   = $total_voxels,
                a.status         = 'completed',
                a.fem_cycle_timestamps =
                    CASE WHEN size(coalesce(a.fem_cycle_timestamps, [])) >= 2000
                         THEN coalesce(a.fem_cycle_timestamps, [])[1..] + [$timestamp]
                         ELSE coalesce(a.fem_cycle_timestamps, []) + [$timestamp]
                    END
        """, {
            "timestamp": timestamp,
            "sensor_reading": sensor_reading.tolist(),
            "total_voxels": len(solid_coords),
        })

        analysis_id = 'latest'
        print(f"✅ Merged FEMAnalysis node 'latest' at {timestamp}")

        # Overwrite per-voxel FEM scalars (no list append → no unbounded growth)
        batch_size = 1000
        for batch_start in range(0, len(solid_coords), batch_size):
            batch_end = min(batch_start + batch_size, len(solid_coords))

            batch_data = []
            for idx in range(batch_start, batch_end):
                coord = solid_coords[idx]
                i, j, k = coord
                batch_data.append({
                    "grid_i":          int(i),
                    "grid_j":          int(j),
                    "grid_k":          int(k),
                    "eps_xx":          float(strain_solid[0, idx]),
                    "eps_yy":          float(strain_solid[1, idx]),
                    "eps_zz":          float(strain_solid[2, idx]),
                    "sigma_xx":        float(stress_solid[0, idx]),
                    "sigma_yy":        float(stress_solid[1, idx]),
                    "sigma_zz":        float(stress_solid[2, idx]),
                    "sigma_xy":        float(stress_solid[3, idx]),
                    "sigma_yz":        float(stress_solid[4, idx]),
                    "sigma_xz":        float(stress_solid[5, idx]),
                    "stress_magnitude": float(stress_magnitude[idx]),
                })
                updated_voxels.append({
                    "grid_position": [int(i), int(j), int(k)],
                    "stress_magnitude": float(stress_magnitude[idx]),
                })

            # Plain SET overwrites — each cycle refreshes values, nothing accumulates
            session.run("""
                UNWIND $batch AS data
                MATCH (v:Voxel {
                    grid_i: data.grid_i,
                    grid_j: data.grid_j,
                    grid_k: data.grid_k
                })
                SET v.eps_xx           = data.eps_xx,
                    v.eps_yy           = data.eps_yy,
                    v.eps_zz           = data.eps_zz,
                    v.sigma_xx         = data.sigma_xx,
                    v.sigma_yy         = data.sigma_yy,
                    v.sigma_zz         = data.sigma_zz,
                    v.sigma_xy         = data.sigma_xy,
                    v.sigma_yz         = data.sigma_yz,
                    v.sigma_xz         = data.sigma_xz,
                    v.stress_magnitude = data.stress_magnitude,
                    v.fem_timestamp    = $timestamp,
                    v.last_updated     = $timestamp
                RETURN count(v) AS updated_count
            """, {"batch": batch_data, "timestamp": timestamp})

            if batch_end % 1000 == 0 or batch_end == len(solid_coords):
                print(f"   Updated {batch_end}/{len(solid_coords)} voxels...")

        print(f"✅ Overwrote {len(updated_voxels)} voxels with latest FEM results")

        # Update summary stats on the single FEMAnalysis node
        session.run("""
            MATCH (a:FEMAnalysis {analysis_id: 'latest'})
            MATCH (v:Voxel) WHERE v.stress_magnitude IS NOT NULL
            WITH a,
                 avg(v.stress_magnitude) AS avg_s,
                 max(v.stress_magnitude) AS max_s,
                 min(v.stress_magnitude) AS min_s,
                 count(v)                AS cnt
            SET a.avg_stress_magnitude = avg_s,
                a.max_stress_magnitude = max_s,
                a.min_stress_magnitude = min_s,
                a.result_count         = cnt
        """)
        print("✅ Updated FEMAnalysis summary statistics")
    
    driver.close()
    
    summary = {
        "success": True,
        "analysis_id": "latest",
        "timestamp": timestamp,
        "voxels_updated": len(updated_voxels),
        "avg_stress_magnitude": float(np.mean(stress_magnitude)),
        "max_stress_magnitude": float(np.max(stress_magnitude)),
        "sensor_reading": sensor_reading.tolist()
    }
    
    print(f"\n{'='*50}")
    print("NEO4J UPDATE COMPLETE")
    print(f"{'='*50}")
    print(f"Analysis ID: {analysis_id}")
    print(f"Voxels updated: {len(updated_voxels)}")
    print(f"Avg stress magnitude: {summary['avg_stress_magnitude']:.3e} Pa")
    print(f"Max stress magnitude: {summary['max_stress_magnitude']:.3e} Pa")
    
    return summary


def run_fem_analysis_tool(
    sensor_readings: List[float],
    update_neo4j: bool = True
) -> str:
    """
    Run FEM analysis when sensor data is updated.
    
    This tool ONLY runs when actual sensor readings are provided.
    It does NOT use default values - it requires real sensor data.
    
    Args:
        sensor_readings: REQUIRED list of 4 sensor values in microstrain (e.g., [65.3, 120.1, 5.8, 30.0])
        update_neo4j: Whether to update Neo4j with results (default: True)
        
    Returns:
        JSON string with FEM analysis results and Neo4j update status
    """
    print(f"\n{'='*60}")
    print("FEM ANALYSIS TOOL - SENSOR-DRIVEN")
    print(f"{'='*60}")
    
    # Validate sensor readings are provided
    if sensor_readings is None:
        error_response = {
            "success": False,
            "error": "No sensor readings provided",
            "message": "FEM analysis requires actual sensor data. Please provide 4 sensor readings in microstrain."
        }
        print("❌ Error: No sensor readings provided!")
        return json.dumps(error_response, indent=2)
    
    if len(sensor_readings) != 4:
        error_response = {
            "success": False,
            "error": f"Invalid sensor readings length: {len(sensor_readings)}",
            "message": "FEM analysis requires exactly 4 sensor readings in microstrain."
        }
        print(f"❌ Error: Expected 4 sensor readings, got {len(sensor_readings)}!")
        return json.dumps(error_response, indent=2)
    
    try:
        # Convert sensor readings to proper format
        sensor_array = np.array(sensor_readings).reshape(1, -1)
        print(f"📊 Using sensor readings: {sensor_readings} μE")
        
        # Run the FEM pipeline
        print("\n🔬 Running FEM pipeline...")
        results = run_pipeline(sensor_reading=sensor_array, output_dir="out")
        
        # Convert sensor reading for Neo4j storage
        sensor_for_db = np.array(sensor_readings)  # Already in microstrain
        
        # Update Neo4j if requested
        neo4j_result = None
        if update_neo4j:
            print("\n📊 Updating Neo4j database...")
            neo4j_result = update_neo4j_with_fem_results(
                strain_voxel=results['strain_voxel'],
                stress_voxel=results['stress_voxel'],
                voxel_mask=results['voxel_mask'],
                solid_coords=results['solid_coords'],
                sensor_reading=sensor_for_db
            )
        
        # Build response
        smoothing = results.get("smoothing", {})
        response = {
            "success": True,
            "fem_analysis": {
                "strain_output": results['strain_output_path'],
                "stress_output": results['stress_output_path'],
                "avg_stress_magnitude": float(results['stress_magnitude'].mean()),
                "max_stress_magnitude": float(results['stress_magnitude'].max()),
                "solid_voxels": int(len(results['solid_coords'])),
                "smoothing_mode": smoothing.get("mode", "nn_full"),
                "smoothing_alpha": smoothing.get("alpha", 1.0),
                "sensor_delta_uE": smoothing.get("delta_uE", 0.0),
            },
            "neo4j_update": neo4j_result if update_neo4j else {"skipped": True},
            "sensor_readings": sensor_for_db.tolist(),
            "message": f"FEM analysis completed successfully with sensor readings {sensor_readings} μE"
        }

        if update_neo4j and neo4j_result:
            response["message"] += f". Neo4j updated with {neo4j_result['voxels_updated']} voxel results."
        
        print(f"\n✅ FEM Analysis Tool Complete!")
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e),
            "message": "FEM analysis failed"
        }
        print(f"\n❌ FEM Analysis Tool Error: {e}")
        return json.dumps(error_response, indent=2)


def check_sensor_updates_and_run_fem() -> str:
    """
    Check for recent sensor updates and run FEM analysis if new data is available.
    
    This function should be called periodically to check if sensors have new data
    and automatically trigger FEM analysis when sensor readings are updated.
    
    Returns:
        JSON string with status and results
    """
    print(f"\n{'='*60}")
    print("SENSOR UPDATE CHECKER & FEM TRIGGER")
    print(f"{'='*60}")
    
    try:
        from datetime import datetime, timedelta
        
        # Check for recent sensor updates (last 5 minutes)
        driver = _get_neo4j_driver()
        conn_info = Settings.get_connection_info()
        
        with driver.session(database=conn_info["database"]) as session:
            # Look for recent sensor updates
            cutoff_time = (datetime.now() - timedelta(minutes=5)).isoformat()
            
            # Check for voxels with recent sensor data updates
            query = """
                MATCH (v:Voxel)
                WHERE v.last_sensor_update > $cutoff_time
                AND (v.strain_uE IS NOT NULL OR v.temp_c IS NOT NULL OR v.load_N IS NOT NULL)
                RETURN v.id, v.strain_uE, v.temp_c, v.load_N, v.last_sensor_update
                ORDER BY v.last_sensor_update DESC
                LIMIT 10
            """
            
            results = session.run(query, {"cutoff_time": cutoff_time})
            sensor_updates = list(results)
            
            if not sensor_updates:
                return json.dumps({
                    "success": True,
                    "status": "no_updates",
                    "message": "No recent sensor updates found",
                    "cutoff_time": cutoff_time
                }, indent=2)
            
            print(f"📊 Found {len(sensor_updates)} voxels with recent sensor updates")
            
            # Extract sensor readings (use strain_uE as primary sensor)
            sensor_readings = []
            for record in sensor_updates:
                strain = record.get("strain_uE")
                if strain is not None:
                    sensor_readings.append(float(strain))
            
            if len(sensor_readings) < 4:
                return json.dumps({
                    "success": True,
                    "status": "insufficient_data",
                    "message": f"Only {len(sensor_readings)} sensor readings available, need 4",
                    "available_readings": sensor_readings
                }, indent=2)
            
            # Take first 4 sensor readings
            sensor_readings = sensor_readings[:4]
            print(f"📊 Using sensor readings: {sensor_readings} μE")
            
            # Run FEM analysis with these sensor readings
            fem_result = run_fem_analysis_tool(sensor_readings, update_neo4j=True)
            fem_data = json.loads(fem_result)
            
            return json.dumps({
                "success": True,
                "status": "fem_triggered",
                "message": "Sensor updates detected, FEM analysis triggered",
                "sensor_readings": sensor_readings,
                "fem_analysis": fem_data
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": "Failed to check sensor updates"
        }, indent=2)


def reset_smoothing_state():
    """
    Clear the temporal-smoothing caches.

    Call this when:
    - The simulation is stopped and restarted (so the first new cycle runs
      a full NN inference rather than blending against a stale field).
    - The voxel model is reloaded or changed.
    - Unit tests need a clean slate.

    Does NOT unload the model weights — those stay in memory for fast reuse.
    """
    global _prev_stress_voxel, _prev_strain_voxel, _prev_sensor_reading
    _prev_stress_voxel   = None
    _prev_strain_voxel   = None
    _prev_sensor_reading = None
    print("🔄 [FEM] Temporal-smoothing state cleared")


# Create Agno Toolkit for FEM Analysis
fem_toolkit = Toolkit(
    name="fem_analysis",
    tools=[run_fem_analysis_tool, check_sensor_updates_and_run_fem]
)


# Export for easy import
__all__ = [
    "fem_toolkit",
    "run_fem_analysis_tool",
    "check_sensor_updates_and_run_fem",
    "run_pipeline",
    "update_neo4j_with_fem_results",
    "reset_smoothing_state",
]


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Structural Health Monitoring Pipeline")
    parser.add_argument("--sensors", nargs=4, type=float, 
                       help="4 sensor readings in microstrain (e.g., --sensors 65.3 120.1 5.8 30.0)")
    parser.add_argument("--output-dir", default="out", 
                       help="Output directory for results (default: out)")
    parser.add_argument("--device", choices=['auto', 'cpu', 'cuda'], default='auto',
                       help="Device to use for computation")
    
    args = parser.parse_args()
    
    # Override device if specified
    if args.device != 'auto':
        global device
        device = torch.device(args.device)
    
    # Run pipeline
    try:
        results = run_pipeline(
            sensor_reading=args.sensors,
            output_dir=args.output_dir
        )
        print(f"\nResults saved to: {args.output_dir}/")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Make sure you're running from the correct directory with 'function/root/' folder")
    except Exception as e:
        print(f"Error running pipeline: {e}")


if __name__ == "__main__":
    main()
