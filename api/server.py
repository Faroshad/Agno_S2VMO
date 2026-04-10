#!/usr/bin/env python3
"""
S2VMO Digital Twin – FastAPI Backend
Connects the UI to MQTT, FEM simulation, Neo4j, and GraphRAG chatbot.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── Project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("s2vmo.api")

# ─── Lazy project imports (graceful degradation when deps missing) ─────────────
def _try_import(module: str, attr: str = None):
    try:
        mod = __import__(module, fromlist=[attr] if attr else [])
        return getattr(mod, attr) if attr else mod
    except Exception as e:
        logger.warning(f"Optional import {module}.{attr} unavailable: {e}")
        return None


# ─── App State ────────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        # MQTT
        self.mqtt_connected = False
        self.mqtt_client = None
        self.mqtt_messages: List[Dict] = []
        self.mqtt_config = {
            "broker": "broker.hivemq.com",
            "port": 1883,
            "topic": "farshad/s2vmo/dome/stream",
        }

        # Simulation
        self.sim_running = False
        self.sim_cycle = 0
        self.sim_last_stats: Optional[Dict] = None
        self._stop_sim = threading.Event()

        # Neo4j
        self.neo4j_connected = False

        # Real-time
        self._ws_clients: List[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Data cache
        self.sensor_history: Dict[str, List] = {
            "timestamps": [], "S1": [], "S2": [], "S3": [], "S4": [],
        }
        self.alerts: List[Dict] = []
        self.chat_history: List[Dict] = []

        # RAG / chat agent
        self._rag_agent = None
        self._rag_lock = threading.Lock()
        self.agent_ready: bool = False   # True once GraphRAG initialised successfully
        self.agent_error: str = ""       # Set if init fails

        # FEM model readiness (checked on startup)
        self.fem_model_ready: bool = False

        # Real-time MQTT sensor data (from physical model)
        self.mqtt_last_sensor_values: Optional[List[float]] = None
        self.mqtt_last_rich_sensors: Optional[Dict[str, Dict]] = None  # full acc/gyro/strain per sensor
        self.mqtt_new_data = threading.Event()   # set when new sensor reading arrives
        self.sim_interval: float = 5.0           # seconds between FEM cycles

        # Accumulation buffers — filled each MQTT message, drained + averaged each FEM cycle
        self.mqtt_readings_buffer: List[List[float]] = []   # [N x 4] readings accumulated this interval
        self.mqtt_rich_buffer: List[Dict] = []              # corresponding rich sensor dicts
        self._buffer_lock = threading.Lock()                # protects the two buffers above

        # Sensor assignment — set via UI before pipeline can run
        # Dict: {"S1": {"grid_i": int, "grid_j": int, "grid_k": int, "sensor_type": str}, ...}
        self.sensor_positions: Dict[str, Dict] = {}
        self.sensors_assigned: bool = False


state = AppState()

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="S2VMO Digital Twin", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = PROJECT_ROOT / "ui"
UI_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(UI_DIR / "index.html"))


# ─── Startup / Shutdown ───────────────────────────────────────────────────────
def _fem_probe():
    """Check that all trained model files exist and mark state accordingly."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    required = [
        root / "models" / "s2v_model_final.pth",
        root / "models" / "unet_model_final.pth",
        root / "models" / "strain_mean.npy",
        root / "models" / "strain_std.npy",
        root / "out"    / "voxel_grid.npz",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.warning(f"FEM model: missing files – {missing}")
        state.fem_model_ready = False
    else:
        state.fem_model_ready = True
        logger.info("FEM model: all files present ✓")
    _broadcast_sync("fem_ready", {"ready": state.fem_model_ready, "missing": missing})


def _agent_warmup():
    """Initialise the GraphRAG agent once at startup (runs in background thread)."""
    try:
        agent = _get_rag_agent()      # creates & caches GraphRAGAgent
        state.agent_ready = True
        state.agent_error = ""
        logger.info("GraphRAG agent: ready ✓")
        _broadcast_sync("agent_ready", {"ready": True, "error": ""})
    except Exception as exc:
        state.agent_ready = False
        state.agent_error = str(exc)
        logger.warning(f"GraphRAG agent: init failed – {exc}")
        _broadcast_sync("agent_ready", {"ready": False, "error": str(exc)})


async def _neo4j_reconnect_watchdog():
    """If Neo4j was down at startup, retry every 30s and notify WebSocket clients when it connects."""
    while True:
        await asyncio.sleep(30)
        if state.neo4j_connected:
            continue
        try:
            from tools.neo4j_tools import neo4j_ping
            if neo4j_ping():
                state.neo4j_connected = True
                logger.info("Neo4j: came online (watchdog)")
                _broadcast_sync("neo4j_status", {"connected": True})
        except Exception:
            pass


@app.on_event("startup")
async def on_startup():
    state._loop = asyncio.get_event_loop()

    # ── Probe Neo4j (retry: DB often starts after the API process) ─────────
    try:
        from tools.neo4j_tools import neo4j_ping
        _neo4j_ok = False
        for attempt in range(1, 6):
            if neo4j_ping():
                _neo4j_ok = True
                state.neo4j_connected = True
                logger.info("Neo4j: connected")
                break
            logger.warning(f"Neo4j: attempt {attempt}/5 not reachable")
            if attempt < 5:
                time.sleep(2.0)
        if not _neo4j_ok:
            state.neo4j_connected = False
            logger.warning(
                "Neo4j: still offline after retries — start Neo4j Desktop/Docker, "
                "or set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env"
            )
    except Exception as e:
        logger.warning(f"Neo4j: probe failed – {e}")

    # ── Probe FEM model files ───────────────────────────────────
    _fem_probe()

    # ── Warm up GraphRAG agent in background ───────────────────
    threading.Thread(target=_agent_warmup, daemon=True, name="agent-warmup").start()

    # ── Periodically retry Neo4j if it was offline at startup (user starts DB later) ─
    asyncio.create_task(_neo4j_reconnect_watchdog())


# ─── Broadcast helper ─────────────────────────────────────────────────────────
async def _broadcast(event_type: str, data: Any):
    msg = json.dumps({"type": event_type, "data": data, "ts": datetime.now().isoformat()},
                     default=str)
    dead = []
    for ws in state._ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in state._ws_clients:
            state._ws_clients.remove(ws)


def _broadcast_sync(event_type: str, data: Any):
    """Thread-safe broadcast from non-async code."""
    if state._loop:
        asyncio.run_coroutine_threadsafe(_broadcast(event_type, data), state._loop)


# ─── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    state._ws_clients.append(websocket)
    # Send current system state immediately
    await websocket.send_text(json.dumps({
        "type": "state_snapshot",
        "data": {
            "mqtt_connected":   state.mqtt_connected,
            "sim_running":      state.sim_running,
            "neo4j_connected":  state.neo4j_connected,
            "sim_cycle":        state.sim_cycle,
            "sim_interval":     state.sim_interval,
            "sensor_history":   state.sensor_history,
            "sensors_assigned": state.sensors_assigned,
            "fem_model_ready":  state.fem_model_ready,
            "agent_ready":      state.agent_ready,
            "agent_error":      state.agent_error,
        },
    }, default=str))
    try:
        while True:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=45.0)
            if raw == "ping":
                await websocket.send_text("pong")
    except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
        pass
    finally:
        if websocket in state._ws_clients:
            state._ws_clients.remove(websocket)


# ═══════════════════════════════════════════════════════════════════════════════
# MQTT
# ═══════════════════════════════════════════════════════════════════════════════

def _mqtt_on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        state.mqtt_connected = True
        client.subscribe(state.mqtt_config["topic"])
        _broadcast_sync("mqtt_status", {"connected": True, "topic": state.mqtt_config["topic"]})
        logger.info(f"MQTT connected → {state.mqtt_config['topic']}")
    else:
        logger.warning(f"MQTT connect failed: rc={reason_code}")


def _parse_sensor_array(payload_list: list) -> Optional[Dict]:
    """
    Parse the physical-device JSON array format:
      [ {"id":1,"type":"structural","hx711_raw":-15294,"strain_uE":-1500.0,
         "acc_g":{"x":0.056,"y":-0.038,"z":1.062},
         "gyro_dps":{"x":-6.47,"y":-0.65,"z":0.35}},
        {"id":2, ...} ]

    Returns a dict with:
      "fem_input":  [S1, S2, S3, S4]  — strain_uE values for FEM
      "rich":       { "S1": {...}, "S2": {...}, "S3": {...}, "S4": {...} }
    Physical mapping:
      sensor id=1  →  S1 (MPU voxel) + S3 (SG voxel)
      sensor id=2  →  S2 (MPU voxel) + S4 (SG voxel)
    """
    by_id: Dict[int, dict] = {}
    for item in payload_list:
        if isinstance(item, dict) and "id" in item:
            by_id[int(item["id"])] = item

    if not (1 in by_id and 2 in by_id):
        return None

    def _rich(item: dict, label: str) -> dict:
        ag = item.get("acc_g") or {}
        gd = item.get("gyro_dps") or {}
        return {
            "sensor_id":   label,
            "strain_uE":   float(item.get("strain_uE", 0)),
            "hx711_raw":   int(item.get("hx711_raw", 0)),
            "acc_x":       float(ag.get("x", 0)),
            "acc_y":       float(ag.get("y", 0)),
            "acc_z":       float(ag.get("z", 0)),
            "gyro_x":      float(gd.get("x", 0)),
            "gyro_y":      float(gd.get("y", 0)),
            "gyro_z":      float(gd.get("z", 0)),
        }

    s1_data = _rich(by_id[1], "S1")   # MPU voxel — sensor node 1
    s2_data = _rich(by_id[2], "S2")   # MPU voxel — sensor node 2
    s3_data = _rich(by_id[1], "S3")   # SG  voxel — sensor node 1
    s3_data["sensor_id"] = "S3"
    s4_data = _rich(by_id[2], "S4")   # SG  voxel — sensor node 2
    s4_data["sensor_id"] = "S4"

    return {
        "fem_input": [
            s1_data["strain_uE"],
            s2_data["strain_uE"],
            s3_data["strain_uE"],
            s4_data["strain_uE"],
        ],
        "rich": {"S1": s1_data, "S2": s2_data, "S3": s3_data, "S4": s4_data},
    }


def _extract_sensor_values(payload) -> Optional[List[float]]:
    """
    Extract [S1, S2, S3, S4] strain values from an MQTT payload (any format).
    Primary format: JSON array from physical device (see _parse_sensor_array).
    Falls back to legacy dict formats for compatibility.
    """
    # ── Primary format: array from physical device ──────────────────────────
    if isinstance(payload, list):
        parsed = _parse_sensor_array(payload)
        if parsed:
            return parsed["fem_input"]
        return None

    if not isinstance(payload, dict):
        return None

    # ── Legacy dict formats ─────────────────────────────────────────────────

    # {"fem_input": [s1, s2, s3, s4]}
    if "fem_input" in payload and isinstance(payload["fem_input"], list):
        vals = payload["fem_input"]
        if len(vals) >= 4:
            return [float(v) for v in vals[:4]]

    # {"sensors": {"S1": {"strain_uE": x}, ...}}
    if "sensors" in payload and isinstance(payload["sensors"], dict):
        sensors = payload["sensors"]
        result = []
        for key in ["S1", "S2", "S3", "S4"]:
            if key in sensors:
                s = sensors[key]
                val = s.get("strain_uE", s.get("value", s.get("strain", 0))) if isinstance(s, dict) else float(s)
                result.append(float(val))
        if len(result) == 4:
            return result

    # {"S1": x, "S2": x, "S3": x, "S4": x}
    result = [float(payload[k]) for k in ["S1", "S2", "S3", "S4"] if k in payload]
    if len(result) == 4:
        return result

    # {"strain": [...]} / {"readings": [...]}
    for field in ("strain", "readings", "values", "data"):
        v = payload.get(field)
        if isinstance(v, list) and len(v) >= 4:
            return [float(x) for x in v[:4]]

    return None


def _mqtt_on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        payload = msg.payload.decode("utf-8", errors="replace")
    entry = {"ts": datetime.now().isoformat(), "topic": msg.topic, "payload": payload}
    state.mqtt_messages.append(entry)
    if len(state.mqtt_messages) > 200:
        state.mqtt_messages = state.mqtt_messages[-200:]

    # Parse and store sensor readings from the physical model
    parsed_rich: Optional[Dict] = None
    if isinstance(payload, list):
        # Primary format: JSON array from physical device
        parsed = _parse_sensor_array(payload)
        if parsed:
            state.mqtt_last_rich_sensors = parsed["rich"]
            state.mqtt_last_sensor_values = parsed["fem_input"]
            parsed_rich = parsed["rich"]
            # Accumulate into interval buffer (averaged before next FEM cycle)
            with state._buffer_lock:
                state.mqtt_readings_buffer.append(parsed["fem_input"])
                state.mqtt_rich_buffer.append(parsed["rich"])
            state.mqtt_new_data.set()
            logger.debug(f"MQTT array parsed: strain values={parsed['fem_input']} (buffer size={len(state.mqtt_readings_buffer)})")
            # Broadcast live sensor values so the UI charts update immediately
            _broadcast_sync("mqtt_sensor_update", {
                "readings": parsed["fem_input"],
                "rich": {
                    sid: {
                        "strain_uE": d["strain_uE"],
                        "hx711_raw": d["hx711_raw"],
                        "acc_g":     {"x": d["acc_x"], "y": d["acc_y"], "z": d["acc_z"]},
                        "gyro_dps":  {"x": d["gyro_x"], "y": d["gyro_y"], "z": d["gyro_z"]},
                    }
                    for sid, d in parsed["rich"].items()
                },
                "ts": entry["ts"],
            })
    else:
        vals = _extract_sensor_values(payload)
        if vals:
            state.mqtt_last_sensor_values = vals
            with state._buffer_lock:
                state.mqtt_readings_buffer.append(vals)
            state.mqtt_new_data.set()
            logger.debug(f"MQTT sensor readings extracted: {vals}")

    # ── Write sensor data to Neo4j on every MQTT message ──────────────────────
    # Runs even when simulation is stopped so the graph always has live readings.
    # Guard: sensors must be assigned and Neo4j must be online.
    if state.sensors_assigned and state.sensor_positions and state.mqtt_last_sensor_values:
        try:
            from tools.sensor_update import update_sensor_readings
            update_sensor_readings(
                state.mqtt_last_sensor_values,
                sensor_positions=state.sensor_positions,
                timestamp=entry["ts"],
                rich_sensors=parsed_rich or state.mqtt_last_rich_sensors,
            )
            logger.debug("Neo4j sensor update written from MQTT message")
        except Exception as _neo_exc:
            logger.debug(f"Neo4j sensor write skipped: {_neo_exc}")

    _broadcast_sync("mqtt_message", entry)


def _mqtt_on_disconnect(client, userdata, rc, *args):
    state.mqtt_connected = False
    _broadcast_sync("mqtt_status", {"connected": False})


class MQTTConfigModel(BaseModel):
    broker: str = "broker.hivemq.com"
    port: int = 1883
    topic: str = "farshad/s2vmo/dome/stream"


@app.post("/api/mqtt/start")
async def mqtt_start(config: Optional[MQTTConfigModel] = None):
    if state.mqtt_connected:
        return {"status": "already_connected"}
    if config:
        state.mqtt_config.update(config.model_dump())

    # Wipe previous sensor readings and FEM results so this session starts clean.
    # Voxel geometry / neighbour structure is preserved.
    try:
        from tools.neo4j_tools import reset_sensor_and_fem_data
        reset_result = reset_sensor_and_fem_data()
        logger.info(f"Neo4j reset on MQTT connect: {reset_result}")
    except Exception as _reset_exc:
        logger.warning(f"Neo4j reset skipped (Neo4j may be offline): {_reset_exc}")

    try:
        import paho.mqtt.client as mqtt_lib
        client = mqtt_lib.Client(mqtt_lib.CallbackAPIVersion.VERSION2)
        client.on_connect = _mqtt_on_connect
        client.on_message = _mqtt_on_message
        client.on_disconnect = _mqtt_on_disconnect
        state.mqtt_client = client

        def _run():
            try:
                client.connect(state.mqtt_config["broker"], state.mqtt_config["port"], 60)
                client.loop_forever()
            except Exception as e:
                logger.error(f"MQTT thread: {e}")
                state.mqtt_connected = False

        threading.Thread(target=_run, daemon=True, name="mqtt-loop").start()
        return {"status": "connecting", "config": state.mqtt_config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mqtt/stop")
async def mqtt_stop():
    if state.mqtt_client:
        state.mqtt_client.disconnect()
        state.mqtt_client = None
    state.mqtt_connected = False
    await _broadcast("mqtt_status", {"connected": False})
    return {"status": "disconnected"}


@app.get("/api/mqtt/status")
async def mqtt_status():
    return {
        "connected": state.mqtt_connected,
        "config": state.mqtt_config,
        "message_count": len(state.mqtt_messages),
        "recent": state.mqtt_messages[-10:],
    }


@app.get("/api/mqtt/messages")
async def mqtt_messages(limit: int = 50):
    return state.mqtt_messages[-limit:]


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _sim_worker(stop_event: threading.Event):
    """Background simulation thread – MQTT physical sensor → FEM → Neo4j loop."""
    # Core pipeline imports — if these fail the worker cannot run
    try:
        from tools.sensor_update import update_sensor_readings
        from tools.fem_tool import run_pipeline, update_neo4j_with_fem_results, reset_smoothing_state
    except ImportError as e:
        logger.error(f"Sim worker: core import failed — {e}")
        state.sim_running = False
        _broadcast_sync("sim_error", {"cycle": 0, "error": f"Core import failed: {e}"})
        return

    # Clear any stale EMA / gating state left over from a previous run
    try:
        reset_smoothing_state()
    except Exception:
        pass

    # Optional monitoring agent — missing file must NOT prevent the pipeline from running
    monitoring = None
    try:
        from agents.monitoring_agent import MonitoringAgent
        monitoring = MonitoringAgent()
    except Exception as e:
        logger.warning(f"Sim worker: MonitoringAgent unavailable (continuing without it) — {e}")

    state.sim_cycle = 0
    _broadcast_sync("sim_status", {"running": True})
    logger.info("Sim worker started — accumulating MQTT readings each interval before FEM")

    while not stop_event.is_set():
        cycle_t0 = time.time()

        # ── 1. Wait sim_interval while MQTT readings accumulate in the buffer ─
        # The buffer (_mqtt_readings_buffer) is filled on every MQTT message,
        # even while FEM is computing.  Draining it here therefore captures
        # ALL readings since the PREVIOUS drain:
        #   • readings that arrived while last cycle's FEM was running
        #   • readings that arrived while Neo4j was being written
        #   • readings that arrived during this wait period
        # This guarantees zero sensor data is lost between FEM cycles,
        # regardless of how long the NN inference takes.
        interval = max(state.sim_interval, 1.0)
        stop_event.wait(timeout=interval)

        if stop_event.is_set():
            break

        # ── 2. Drain accumulation buffer and compute average ─────────────
        with state._buffer_lock:
            buf = list(state.mqtt_readings_buffer)
            rich_buf = list(state.mqtt_rich_buffer)
            state.mqtt_readings_buffer.clear()
            state.mqtt_rich_buffer.clear()

        ts = datetime.now().isoformat()
        if buf:
            avg_values = list(np.mean(buf, axis=0))
            n = len(buf)
            source = f"MQTT avg ({n} reading{'s' if n != 1 else ''})"
            rich_sensors = rich_buf[-1] if rich_buf else state.mqtt_last_rich_sensors
        else:
            # No MQTT readings buffered this interval — check for any previous reading
            avg_values, source = _current_sensor_values()
            rich_sensors = state.mqtt_last_rich_sensors

        # Skip the cycle entirely if there is no real sensor data
        if avg_values is None:
            logger.warning("Sim: no MQTT sensor data yet — skipping FEM cycle (waiting for sensors)")
            _broadcast_sync("sim_error", {
                "cycle": state.sim_cycle,
                "error": "Waiting for MQTT sensor data — connect your physical sensors first",
            })
            continue

        sensor_values = avg_values

        # Signal UI: FEM cycle is starting (flashes the FEM node active)
        _broadcast_sync("fem_cycle_start", {"cycle": state.sim_cycle + 1})
        logger.info(f"Sim cycle {state.sim_cycle+1}: source={source} values={sensor_values}")

        try:
            # 2. Push to Neo4j sensor nodes (using UI-assigned positions + rich IMU data)
            if state.sensors_assigned and state.sensor_positions:
                update_sensor_readings(
                    sensor_values,
                    sensor_positions=state.sensor_positions,
                    timestamp=ts,
                    rich_sensors=rich_sensors,
                )

            # 3. FEM pipeline
            arr = np.array(sensor_values).reshape(1, -1)
            fem = run_pipeline(sensor_reading=arr, output_dir="out")

            # 4. Write FEM results to Neo4j
            neo4j_res = update_neo4j_with_fem_results(
                strain_voxel=fem["strain_voxel"],
                stress_voxel=fem["stress_voxel"],
                voxel_mask=fem["voxel_mask"],
                solid_coords=fem["solid_coords"],
                sensor_reading=np.array(sensor_values),
                timestamp=ts,
            )

            # 5. Monitoring
            monitoring_info: Dict = {}
            if monitoring:
                try:
                    monitoring_info = monitoring.assess_cycle(
                        cycle=state.sim_cycle, timestamp=ts
                    )
                except Exception:
                    pass

            # Update local sensor history
            state.sensor_history["timestamps"].append(ts)
            for i, key in enumerate(["S1", "S2", "S3", "S4"]):
                state.sensor_history[key].append(float(sensor_values[i]))
            # Keep last 120 readings
            if len(state.sensor_history["timestamps"]) > 120:
                for k in state.sensor_history:
                    state.sensor_history[k] = state.sensor_history[k][-120:]

            avg_stress = float(fem["stress_magnitude"].mean())
            max_stress = float(fem["stress_magnitude"].max())
            smoothing  = fem.get("smoothing", {})
            state.sim_cycle += 1
            stats = {
                "cycle": state.sim_cycle,
                "sensor_readings": [float(v) for v in sensor_values],
                "avg_stress_pa": avg_stress,
                "max_stress_pa": max_stress,
                "voxels_updated": neo4j_res.get("voxels_updated", 0),
                "timestamp": ts,
                "duration_s": round(time.time() - cycle_t0, 2),
                "monitoring": monitoring_info,
                "source": source,
                # Smoothing diagnostics — helps the UI and logs show whether
                # the FEM was re-computed, gated (reused), or EMA-blended.
                "fem_mode":    smoothing.get("mode",     "nn_full"),
                "fem_alpha":   smoothing.get("alpha",    1.0),
                "sensor_delta_uE": smoothing.get("delta_uE", 0.0),
                "mqtt_samples": len(buf) if buf else 0,
            }
            state.sim_last_stats = stats
            logger.info(
                f"Cycle {state.sim_cycle} complete | "
                f"mode={stats['fem_mode']} alpha={stats['fem_alpha']:.2f} "
                f"delta={stats['sensor_delta_uE']:.1f}μE "
                f"mqtt_n={stats['mqtt_samples']} "
                f"avg={avg_stress:.2e}Pa max={max_stress:.2e}Pa"
            )
            _broadcast_sync("sim_cycle", stats)

            # Broadcast updated Neo4j counts so the UI node refreshes immediately
            try:
                from tools.neo4j_tools import _execute_query
                counts = _get_neo4j_counts(_execute_query)
                _broadcast_sync("neo4j_updated", counts)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Sim cycle {state.sim_cycle} error: {e}")
            _broadcast_sync("sim_error", {"cycle": state.sim_cycle, "error": str(e)})

        elapsed = time.time() - cycle_t0
        # Already waited sim_interval above; just drain any leftover time
        stop_event.wait(0.1)

    state.sim_running = False
    if monitoring:
        try:
            monitoring.close()
        except Exception:
            pass
    _broadcast_sync("sim_status", {"running": False})


def _current_sensor_values() -> Tuple[Optional[List[float]], str]:
    """Return (sensor_values, source).

    Returns (None, reason) when no MQTT data has been received yet.
    FEM must not run on fabricated default values — only on real sensor data.
    """
    if state.mqtt_last_sensor_values is not None:
        return list(state.mqtt_last_sensor_values), "MQTT (physical model)"
    return None, "no MQTT data — connect sensors first"


@app.post("/api/sim/start")
async def sim_start():
    if state.sim_running:
        return {"status": "already_running", "cycle": state.sim_cycle}
    state.sim_running = True
    state._stop_sim.clear()
    threading.Thread(target=_sim_worker, args=(state._stop_sim,),
                     daemon=True, name="sim-loop").start()
    await _broadcast("sim_status", {"running": True})
    src = "MQTT (physical sensors)" if state.mqtt_connected else "waiting for MQTT sensor data"
    return {"status": "started", "data_source": src}


@app.post("/api/sim/stop")
async def sim_stop():
    state._stop_sim.set()
    state.sim_running = False
    # Clear temporal-smoothing caches so the next start begins with a fresh
    # first-cycle NN inference rather than blending against a stale field.
    try:
        from tools.fem_tool import reset_smoothing_state
        reset_smoothing_state()
    except Exception:
        pass
    await _broadcast("sim_status", {"running": False})
    return {"status": "stopped"}


class SimIntervalModel(BaseModel):
    interval_s: float

@app.put("/api/sim/interval")
async def set_sim_interval(body: SimIntervalModel):
    state.sim_interval = max(1.0, float(body.interval_s))
    return {"interval_s": state.sim_interval}


@app.get("/api/mqtt/sensor_values")
async def mqtt_sensor_values():
    """Return the latest sensor readings extracted from MQTT messages."""
    return {
        "available": state.mqtt_last_sensor_values is not None,
        "values": state.mqtt_last_sensor_values,
        "labels": ["S1", "S2", "S3", "S4"],
        "unit": "μE (microstrain)",
        "source": "MQTT physical model",
    }


@app.post("/api/sim/step")
async def sim_step():
    """Single-shot FEM cycle using live MQTT sensor data. Refuses to run without real data."""
    def _run_one():
        try:
            from tools.sensor_update import update_sensor_readings
            from tools.fem_tool import run_pipeline, update_neo4j_with_fem_results

            sensor_values, source = _current_sensor_values()
            if sensor_values is None:
                raise RuntimeError(
                    "No MQTT sensor data available — connect your physical sensors "
                    "and wait for at least one reading before running a FEM step."
                )
            ts = datetime.now().isoformat()

            # Push to Neo4j sensor nodes only if positions are assigned
            if state.sensors_assigned and state.sensor_positions:
                update_sensor_readings(
                    sensor_values,
                    sensor_positions=state.sensor_positions,
                    timestamp=ts,
                    rich_sensors=state.mqtt_last_rich_sensors,
                )

            # Run the two-stage neural-network FEM pipeline
            arr = np.array(sensor_values).reshape(1, -1)
            fem = run_pipeline(sensor_reading=arr, output_dir="out")
            neo4j_res = update_neo4j_with_fem_results(
                strain_voxel=fem["strain_voxel"],
                stress_voxel=fem["stress_voxel"],
                voxel_mask=fem["voxel_mask"],
                solid_coords=fem["solid_coords"],
                sensor_reading=np.array(sensor_values),
                timestamp=ts,
            )
            avg_stress = float(fem["stress_magnitude"].mean())
            max_stress = float(fem["stress_magnitude"].max())
            state.sim_cycle += 1
            # Update sensor history
            state.sensor_history["timestamps"].append(ts)
            for i, key in enumerate(["S1", "S2", "S3", "S4"]):
                state.sensor_history[key].append(float(sensor_values[i]))
            if len(state.sensor_history["timestamps"]) > 120:
                for k in state.sensor_history:
                    state.sensor_history[k] = state.sensor_history[k][-120:]

            result = {
                "cycle":           state.sim_cycle,
                "sensor_readings": [float(v) for v in sensor_values],
                "avg_stress_pa":   avg_stress,
                "max_stress_pa":   max_stress,
                "voxels_updated":  neo4j_res.get("voxels_updated", 0),
                "duration_s":      round(time.time(), 2),
                "timestamp":       ts,
                "source":          source,
            }
            state.sim_last_stats = result

            # Broadcast neo4j counts update immediately
            try:
                from tools.neo4j_tools import _execute_query
                counts = _get_neo4j_counts(_execute_query)
                _broadcast_sync("neo4j_updated", counts)
            except Exception:
                pass

            _broadcast_sync("sim_cycle", result)
            return result
        except Exception as e:
            raise RuntimeError(str(e))

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run_one)
        await _broadcast("sim_cycle", result)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/sim/status")
async def sim_status_ep():
    # Include smoothing config so clients can show gate / alpha thresholds
    smoothing_cfg = {}
    try:
        from tools.fem_tool import (
            _GATE_THRESHOLD_uE, _ALPHA_MIN, _ALPHA_MAX, _ALPHA_FULL_uE
        )
        smoothing_cfg = {
            "gate_threshold_uE": _GATE_THRESHOLD_uE,
            "alpha_min": _ALPHA_MIN,
            "alpha_max": _ALPHA_MAX,
            "alpha_full_uE": _ALPHA_FULL_uE,
        }
    except Exception:
        pass
    return {
        "running": state.sim_running,
        "cycle": state.sim_cycle,
        "last_stats": state.sim_last_stats,
        "smoothing_config": smoothing_cfg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NEO4J
# ═══════════════════════════════════════════════════════════════════════════════

def _get_neo4j_counts(execute_fn) -> Dict:
    """Return per-label node counts + total relationship count from Neo4j."""
    # Per-label node counts
    label_rows = execute_fn("""
        MATCH (n) UNWIND labels(n) AS lbl
        RETURN lbl, count(n) AS cnt ORDER BY cnt DESC
    """)
    labels: Dict[str, int] = {}
    total_nodes = 0
    for row in (label_rows or []):
        lbl = row.get("lbl", "")
        cnt = int(row.get("cnt", 0))
        labels[lbl] = cnt
        total_nodes += cnt

    # Total relationships
    rel_rows = execute_fn("""
        MATCH ()-[r]->() RETURN type(r) AS rtype, count(r) AS cnt ORDER BY cnt DESC
    """)
    rels: Dict[str, int] = {}
    total_rels = 0
    for row in (rel_rows or []):
        rtype = row.get("rtype", "")
        cnt = int(row.get("cnt", 0))
        rels[rtype] = cnt
        total_rels += cnt

    return {
        "connected":    True,
        "voxel_count":  labels.get("Voxel", 0),
        "fem_analyses": labels.get("FEMAnalysis", 0),
        "sensor_streams": labels.get("SensorStream", 0),
        "total_nodes":  total_nodes,
        "total_rels":   total_rels,
        "labels":       labels,
        "rels":         rels,
        "updated_at":   datetime.now().strftime("%H:%M:%S"),
    }


@app.get("/api/neo4j/status")
async def neo4j_status_ep():
    try:
        from tools.neo4j_tools import neo4j_ping, _execute_query
        if not neo4j_ping():
            state.neo4j_connected = False
            err = "Neo4j unreachable — check that the database is running and .env credentials match"
            _broadcast_sync("neo4j_status", {"connected": False, "error": err})
            return {"connected": False, "error": err}
        counts = _get_neo4j_counts(_execute_query)
        state.neo4j_connected = True
        _broadcast_sync("neo4j_status", {"connected": True})
        return counts
    except Exception as e:
        state.neo4j_connected = False
        _broadcast_sync("neo4j_status", {"connected": False, "error": str(e)})
        return {"connected": False, "error": str(e)}


@app.get("/api/neo4j/sensors/stream")
async def neo4j_sensor_stream():
    try:
        from tools.neo4j_tools import _execute_query
        res = _execute_query("""
            MATCH (s:SensorStream {id: 'global'})
            RETURN s.timestamps AS timestamps,
                   s.S1 AS S1, s.S2 AS S2, s.S3 AS S3, s.S4 AS S4,
                   s.count AS count
        """)
        if res:
            row = res[0]
            return {k: (list(v) if v else []) for k, v in row.items()}
        return {"timestamps": [], "S1": [], "S2": [], "S3": [], "S4": [], "count": 0}
    except Exception:
        return state.sensor_history


@app.get("/api/neo4j/alerts")
async def neo4j_alerts_ep():
    try:
        from tools.neo4j_tools import _execute_query
        res = _execute_query("""
            MATCH (a:StructuralAlert)
            RETURN a.alert_id AS id, a.severity AS severity,
                   a.stress_pa AS stress_pa, a.timestamp AS ts,
                   a.grid_i AS gi, a.grid_j AS gj, a.grid_k AS gk
            ORDER BY a.timestamp DESC LIMIT 30
        """)
        return res
    except Exception as e:
        return []


@app.get("/api/neo4j/fem/latest")
async def neo4j_fem_latest():
    try:
        from tools.neo4j_tools import _execute_query
        res = _execute_query("""
            MATCH (f:FEMAnalysis)
            RETURN f ORDER BY f.timestamp DESC LIMIT 1
        """)
        if res:
            node = res[0].get("f", {})
            return {"available": True, "data": dict(node) if hasattr(node, "items") else node}
        return {"available": False}
    except Exception as e:
        if state.sim_last_stats:
            return {"available": True, "data": state.sim_last_stats}
        return {"available": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# VOXEL DATA
# ═══════════════════════════════════════════════════════════════════════════════

def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode()


@app.get("/api/voxels/geometry")
async def voxels_geometry():
    grid_path = PROJECT_ROOT / "out" / "voxel_grid.npz"
    if not grid_path.exists():
        raise HTTPException(404, "voxel_grid.npz not found – run FEM or initialize Neo4j first")
    try:
        vg = np.load(str(grid_path), allow_pickle=True)
        M: np.ndarray = vg["matrix"].astype(bool)
        origin = vg["origin"].astype(np.float32).reshape(3)
        pitch = float(np.atleast_1d(vg["pitch"])[0])

        solid_idx = np.argwhere(M).astype(np.int16)          # (N, 3)
        world_pos = (solid_idx.astype(np.float32) * pitch + origin)  # (N, 3)

        # Build sensor maps:
        #   sensor_type_map: (i,j,k) -> 1 (MPU) or 2 (SG)
        # MPU sensor type code = 1, SG (strain gauge) = 2
        SENSOR_TYPE_CODES = {"MPU": 1, "SG": 2, "STRAIN_GAUGE": 2, "STRAIN": 2}
        sensor_type_map: dict = {}
        for v in state.sensor_positions.values():
            code = SENSOR_TYPE_CODES.get(v.get("sensor_type", "MPU").upper(), 1)
            sensor_type_map[(v["grid_i"], v["grid_j"], v["grid_k"])] = code
            for vox in (v.get("strut_voxels") or []):
                sensor_type_map[(int(vox[0]), int(vox[1]), int(vox[2]))] = code

        sensor_type_arr = np.array(
            [sensor_type_map.get(tuple(int(x) for x in idx), 0) for idx in solid_idx],
            dtype=np.uint8,
        )
        is_sensor = (sensor_type_arr > 0).astype(np.uint8)

        sensor_info = {
            k: [v["grid_i"], v["grid_j"], v["grid_k"]]
            for k, v in state.sensor_positions.items()
        }

        return {
            "count": int(len(solid_idx)),
            "grid_shape": list(M.shape),
            "origin": origin.tolist(),
            "pitch": pitch,
            "positions_f32_b64": _b64(world_pos.astype(np.float32)),
            "indices_i16_b64": _b64(solid_idx),
            "is_sensor_u8_b64": _b64(is_sensor),
            "sensor_type_u8_b64": _b64(sensor_type_arr),   # 0=none, 1=MPU, 2=SG
            "sensor_info": sensor_info,
            "sensors_assigned": state.sensors_assigned,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Sensor Assignment API ─────────────────────────────────────────────────────

# Voxel grid cached in memory after first load (for strut-finding)
_voxel_solid_set: Optional[set] = None

def _get_solid_set() -> set:
    """Return frozenset of (i,j,k) tuples for all solid voxels (cached)."""
    global _voxel_solid_set
    if _voxel_solid_set is None:
        grid_path = PROJECT_ROOT / "out" / "voxel_grid.npz"
        if grid_path.exists():
            vg = np.load(str(grid_path), allow_pickle=True)
            M = vg["matrix"].astype(bool)
            _voxel_solid_set = set(map(tuple, np.argwhere(M).tolist()))
        else:
            _voxel_solid_set = set()
    return _voxel_solid_set


def _find_strut_voxels(ci: int, cj: int, ck: int, max_walk: int = 12) -> List[tuple]:
    """
    Walk from seed (ci,cj,ck) outward along the dominant strut axis.
    Returns all solid voxels on that strut within max_walk steps.
    For SG sensors: the gauge is bonded to the strut, so we tag every
    voxel along the strut as a sensor to correctly represent the measurement area.
    """
    solid = _get_solid_set()
    seed = (ci, cj, ck)
    if seed not in solid:
        return [seed]   # seed not in grid — return as-is

    axes = [(1,0,0), (0,1,0), (0,0,1)]
    best_run: List[tuple] = [seed]

    for di, dj, dk in axes:
        run = [seed]
        for step in range(1, max_walk + 1):
            pt = (ci + di*step, cj + dj*step, ck + dk*step)
            if pt in solid:
                run.append(pt)
            else:
                break
        for step in range(1, max_walk + 1):
            pt = (ci - di*step, cj - dj*step, ck - dk*step)
            if pt in solid:
                run.append(pt)
            else:
                break
        if len(run) > len(best_run):
            best_run = run

    return sorted(best_run)   # deterministic order


class SensorAssignment(BaseModel):
    id: str          # "S1" … "S4"
    grid_i: int
    grid_j: int
    grid_k: int
    sensor_type: str = "MPU"


@app.post("/api/sensors/assign")
async def assign_sensors(assignments: List[SensorAssignment]):
    """Receive sensor positions from UI, write to Neo4j, unlock pipeline.
    For SG (strain-gauge) sensors the assignment auto-expands to all voxels
    along the strut so the measurement area is correctly represented.
    """
    ids_required = {"S1", "S2", "S3", "S4"}
    ids_provided = {a.id for a in assignments}
    if not ids_required.issubset(ids_provided):
        missing = ids_required - ids_provided
        raise HTTPException(400, f"Missing sensor assignments: {missing}")

    ts = datetime.now().isoformat()
    positions: Dict[str, Dict] = {}

    for a in assignments:
        entry: Dict = {
            "grid_i": a.grid_i, "grid_j": a.grid_j, "grid_k": a.grid_k,
            "sensor_type": a.sensor_type,
        }
        # For strain gauges, expand to full strut so FEM averages correctly
        if a.sensor_type.upper() in ("SG", "STRAIN_GAUGE", "STRAIN"):
            strut = _find_strut_voxels(a.grid_i, a.grid_j, a.grid_k)
            entry["strut_voxels"] = strut   # list of (i,j,k) tuples
            logger.info(f"SG {a.id} at ({a.grid_i},{a.grid_j},{a.grid_k}) "
                        f"expanded to {len(strut)}-voxel strut: {strut}")
        positions[a.id] = entry

    # Persist to Neo4j: clear old sensor tags, then set new ones
    try:
        from tools.neo4j_tools import _execute_query
        _execute_query("MATCH (v:Voxel) WHERE v.is_sensor = true "
                       "REMOVE v.is_sensor, v.sensor_id, v.sensor_type, "
                       "v.is_strut_sensor, v.strut_primary")

        for sid, pos in positions.items():
            stype = pos["sensor_type"]
            strut = pos.get("strut_voxels")

            if strut:
                # Mark every voxel in the strut; primary voxel gets strut_primary=true
                primary = (pos["grid_i"], pos["grid_j"], pos["grid_k"])
                for (vi, vj, vk) in strut:
                    _execute_query("""
                        MERGE (v:Voxel {grid_i: $i, grid_j: $j, grid_k: $k})
                        SET v.is_sensor      = true,
                            v.sensor_id      = $sid,
                            v.sensor_type    = $stype,
                            v.is_strut_sensor = true,
                            v.strut_primary  = $primary,
                            v.last_updated   = $ts
                    """, {"i": vi, "j": vj, "k": vk, "sid": sid, "stype": stype,
                          "primary": (vi,vj,vk) == primary, "ts": ts})
            else:
                _execute_query("""
                    MERGE (v:Voxel {grid_i: $i, grid_j: $j, grid_k: $k})
                    SET v.is_sensor    = true,
                        v.sensor_id   = $sid,
                        v.sensor_type = $stype,
                        v.last_updated = $ts
                """, {"i": pos["grid_i"], "j": pos["grid_j"], "k": pos["grid_k"],
                      "sid": sid, "stype": stype, "ts": ts})
    except Exception as e:
        logger.warning(f"Neo4j sensor assignment write failed: {e}")

    # Update in-memory state (strut_voxels stored as list-of-lists for JSON safety)
    for sid in positions:
        sv = positions[sid].get("strut_voxels")
        if sv:
            positions[sid]["strut_voxels"] = [list(t) for t in sv]

    state.sensor_positions = positions
    state.sensors_assigned = True
    logger.info(f"Sensors assigned: { {k: (v['grid_i'], v['grid_j'], v['grid_k']) for k, v in positions.items()} }")

    await _broadcast("sensors_assigned", {
        "positions": {k: [v["grid_i"], v["grid_j"], v["grid_k"]] for k, v in positions.items()},
        "count": len(positions),
    })

    return {"status": "ok", "sensors_assigned": True, "positions": {
        k: [v["grid_i"], v["grid_j"], v["grid_k"]] for k, v in positions.items()
    }}


@app.get("/api/sensors/assigned")
async def get_assigned_sensors():
    """Return current sensor positions, restoring from Neo4j if in-memory state was lost."""
    # After server restart the in-memory state is empty — try to recover from Neo4j
    if not state.sensors_assigned:
        try:
            from tools.neo4j_tools import _execute_query
            rows = _execute_query(
                """
                MATCH (v:Voxel {is_sensor: true})
                RETURN v.sensor_id AS sid, v.grid_i AS gi, v.grid_j AS gj,
                       v.grid_k AS gk, v.sensor_type AS stype,
                       v.strut_primary AS primary
                """,
                parameters={},
            )
            if rows and len(rows) >= 4:
                recovered: Dict[str, Dict] = {}
                strut_acc: Dict[str, list] = {}
                for row in rows:
                    sid = row.get("sid") or row.get("v.sensor_id")
                    if not sid:
                        continue
                    gi = int(row.get("gi", 0))
                    gj = int(row.get("gj", 0))
                    gk = int(row.get("gk", 0))
                    is_primary = row.get("primary", False)
                    stype = row.get("stype", "")

                    strut_acc.setdefault(sid, []).append([gi, gj, gk])
                    if is_primary or sid not in recovered:
                        recovered[sid] = {
                            "grid_i": gi, "grid_j": gj, "grid_k": gk,
                            "sensor_type": stype,
                        }

                # Attach strut voxels to SG sensors
                for sid in recovered:
                    all_voxels = strut_acc.get(sid, [])
                    if len(all_voxels) > 1:
                        recovered[sid]["strut_voxels"] = all_voxels

                if len(recovered) >= 4:
                    state.sensor_positions = recovered
                    state.sensors_assigned = True
                    logger.info(f"Recovered {len(recovered)} sensor positions from Neo4j")
        except Exception as e:
            logger.debug(f"Neo4j sensor recovery skipped: {e}")

    return {
        "sensors_assigned": state.sensors_assigned,
        "positions": {
            k: [v["grid_i"], v["grid_j"], v["grid_k"]]
            for k, v in state.sensor_positions.items()
        },
    }


@app.get("/api/voxels/stress")
async def voxels_stress():
    stress_path = PROJECT_ROOT / "out" / "sigma_voxel_predicted_pp.npz"
    mask_path = PROJECT_ROOT / "out" / "voxel_mask.npz"
    if not stress_path.exists():
        return {"available": False, "message": "Run simulation to generate stress data"}
    try:
        sdata = np.load(str(stress_path), allow_pickle=True)
        sigma = sdata["sigma_voxel"]                      # (6, I, J, K)

        mdata = np.load(str(mask_path), allow_pickle=True)
        M: np.ndarray = mdata["M"].astype(bool)
        coords = np.argwhere(M)
        i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]

        mag = np.sqrt(sigma[0, i, j, k] ** 2 + sigma[1, i, j, k] ** 2 + sigma[2, i, j, k] ** 2)
        s_min, s_max = float(mag.min()), float(mag.max())
        norm = ((mag - s_min) / (s_max - s_min + 1e-12)).astype(np.float32)

        return {
            "available": True,
            "count": int(len(coords)),
            "min_pa": s_min,
            "max_pa": s_max,
            "stress_norm_f32_b64": _b64(norm),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.get("/api/sensors/history")
async def sensors_history():
    return state.sensor_history


@app.get("/api/fem/status")
async def fem_status_ep():
    """Return FEM model file availability."""
    _fem_probe()   # re-check files in case they changed
    return {
        "ready":   state.fem_model_ready,
        "running": state.sim_running,
        "cycle":   state.sim_cycle,
    }


@app.get("/api/chat/status")
async def chat_status_ep():
    """Return GraphRAG agent readiness."""
    return {
        "ready": state.agent_ready,
        "error": state.agent_error,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CHATBOT (GraphRAG)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_rag_agent():
    with state._rag_lock:
        if state._rag_agent is None:
            from workflows.rag_workflow import RAGWorkflow
            state._rag_agent = RAGWorkflow()
        return state._rag_agent


def _build_live_context() -> str:
    """
    Compact live-state snapshot injected into every agent turn so the LLM
    is aware of the current simulation/sensor state without calling a tool.
    """
    ts = datetime.now().strftime('%H:%M:%S')
    lines = [f"[Live system snapshot — {ts}]"]
    lines.append(
        f"Simulation: {'running' if state.sim_running else 'stopped'} "
        f"· cycle {state.sim_cycle} "
        f"· interval {state.sim_interval}s"
    )
    lines.append(
        f"MQTT: {'connected' if state.mqtt_connected else 'offline'} "
        f"· Neo4j: {'connected' if state.neo4j_connected else 'offline'} "
        f"· FEM model: {'ready' if state.fem_model_ready else 'not loaded'}"
    )
    hist = state.sensor_history
    if hist.get("timestamps"):
        vals = []
        for i, k in enumerate(["S1", "S2", "S3", "S4"]):
            lst = hist.get(k, [])
            vals.append(f"S{i+1}={lst[-1]:.0f}μE" if lst else f"S{i+1}=—")
        lines.append(f"Latest sensor readings: {', '.join(vals)}")
    else:
        lines.append("Latest sensor readings: no data yet")
    if state.alerts:
        lines.append(f"Active structural alerts: {len(state.alerts)}")
    lines.append("[end snapshot]")
    return "\n".join(lines)


class ChatMsg(BaseModel):
    message: str


def _extract_voxels_from_answer(answer: str) -> List[Dict]:
    """
    Parse structured voxel references out of an AI answer string.
    Returns a list of dicts, each with at least one of:
      { x, y, z }          — world coordinates
      { gi, gj, gk }       — grid indices
    """
    import re
    voxels: List[Dict] = []
    seen: set = set()

    def _add(d):
        key = str(sorted(d.items()))
        if key not in seen:
            seen.add(key)
            voxels.append(d)

    # 1. grid_i / grid_j / grid_k  (Cypher RETURN format)
    for m in re.finditer(
        r'grid_i[:\s=]+(-?\d+)[,\s]+grid_j[:\s=]+(-?\d+)[,\s]+grid_k[:\s=]+(-?\d+)',
        answer, re.IGNORECASE
    ):
        _add({"gi": int(m[1]), "gj": int(m[2]), "gk": int(m[3]),
              "label": f"({m[1]},{m[2]},{m[3]})"})

    # 2. x / y / z floats (consecutive, e.g. "x: -4.1, y: -3.9, z: -4.7")
    for m in re.finditer(
        r'x[:\s=]+([-\d.]+)[,;\s]+y[:\s=]+([-\d.]+)[,;\s]+z[:\s=]+([-\d.]+)',
        answer, re.IGNORECASE
    ):
        _add({"x": float(m[1]), "y": float(m[2]), "z": float(m[3]),
              "label": f"({float(m[1]):.2f},{float(m[2]):.2f},{float(m[3]):.2f})"})

    # 3. Parenthesised triples:  (3.9, -3.7, 0.4)  or  (10, 20, 5)
    for m in re.finditer(
        r'\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)',
        answer
    ):
        a, b, c = float(m[1]), float(m[2]), float(m[3])
        # If all three are integers within grid range → grid coords
        if all(v == int(v) and abs(v) < 500 for v in (a, b, c)):
            _add({"gi": int(a), "gj": int(b), "gk": int(c),
                  "label": f"({int(a)},{int(b)},{int(c)})"})
        else:
            _add({"x": a, "y": b, "z": c,
                  "label": f"({a:.2f},{b:.2f},{c:.2f})"})

    return voxels


@app.post("/api/chat")
async def chat_ep(msg: ChatMsg):
    try:
        loop = asyncio.get_event_loop()
        live_ctx = _build_live_context()

        def _run():
            agent = _get_rag_agent()
            result = agent.run(msg.message, save_to_memory=True, live_context=live_ctx)
            return result.get("answer", str(result)) if isinstance(result, dict) else str(result)

        answer = await loop.run_in_executor(None, _run)
        ts = datetime.now().isoformat()
        state.chat_history += [
            {"role": "user",      "content": msg.message, "ts": ts},
            {"role": "assistant", "content": answer,      "ts": ts},
        ]
        voxels = _extract_voxels_from_answer(answer)
        return {"answer": answer, "ts": ts, "voxels": voxels}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/chat/history")
async def chat_history_ep():
    return state.chat_history


@app.post("/api/chat/clear")
async def chat_clear_ep():
    state.chat_history.clear()
    return {"ok": True}


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
