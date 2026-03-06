#!/usr/bin/env python3
"""
Realistic Sensor Generator with Time-Based Environmental Scenarios
Simulates normal conditions with gradual changes and sudden events (e.g., wind gusts)

Scenario:
- Hours 0-1: Normal calm conditions (baseline strain ~50-150 μE)
- Hour 1: Wind starts building gradually (strain increases to ~200-300 μE)
- Hour 1.5: Sudden wind gust (spike to ~500-800 μE)
- Hour 2+: Wind subsides gradually back to normal

Update interval is configurable (default 3 seconds)
"""

import argparse
import json
import math
import random
import sys
import time
from typing import Dict, List, Tuple
import numpy as np

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None


# ================== Fixed Sensor Positions ==================
SENSOR_POSITIONS = {
    "S1": {"grid_i": 20, "grid_j": 15, "grid_k": 40},
    "S2": {"grid_i": 50, "grid_j": 15, "grid_k": 40},
    "S3": {"grid_i": 35, "grid_j": 25, "grid_k": 30},
    "S4": {"grid_i": 65, "grid_j": 10, "grid_k": 50},
}

SENSOR_IDS = ["S1", "S2", "S3", "S4"]
DEFAULT_PERIOD_S = 3.0  # Update interval in seconds

# Environmental scenario parameters
SCENARIO_CONFIG = {
    "baseline_strain": 80.0,  # μE - Normal calm condition
    "baseline_variation": 30.0,  # μE - Natural variation around baseline
    
    # Wind buildup (starts at 1 hour)
    "wind_start_time": 3600.0,  # seconds (1 hour)
    "wind_buildup_duration": 1800.0,  # seconds (30 minutes)
    "wind_max_strain": 300.0,  # μE - Maximum wind-induced strain
    
    # Sudden gust (at 1.5 hours)
    "gust_time": 5400.0,  # seconds (1.5 hours)
    "gust_duration": 120.0,  # seconds (2 minutes)
    "gust_peak_strain": 700.0,  # μE - Peak gust strain
    
    # Wind subsides (after 2 hours)
    "wind_end_time": 7200.0,  # seconds (2 hours)
    "wind_subside_duration": 1800.0,  # seconds (30 minutes)
}

# Sensor-specific phase offsets for vibration
SENSOR_PHASE = {
    "S1": 0.0,
    "S2": math.pi / 2,
    "S3": math.pi,
    "S4": 3 * math.pi / 2
}

# Small structural vibration (always present)
VIBRATION_MODES = [
    {"f_hz": 1.5, "amp_uE": 8.0},   # Low frequency mode
    {"f_hz": 3.8, "amp_uE": 4.0},   # Higher frequency mode
]


class RealisticSensorGenerator:
    """
    Realistic sensor generator with time-based environmental scenarios.
    Simulates normal conditions, wind buildup, sudden gusts, and subsiding.
    """
    
    def __init__(self, seed: int = 42, time_scale: float = 1.0):
        """
        Args:
            seed: Random seed for reproducibility
            time_scale: Time scaling factor (e.g., 0.1 = 10x faster for testing)
        """
        self.rng = random.Random(seed)
        self.t0 = time.time()
        self.seq = 0
        self.time_scale = time_scale
        
        # Initialize sensor-specific profiles
        self.sensor_profiles = {}
        for sensor_id in SENSOR_IDS:
            self.sensor_profiles[sensor_id] = {
                "baseline_offset": self.rng.uniform(-20, 20),  # Individual sensor bias
                "noise_level": self.rng.uniform(2.0, 5.0),  # Sensor noise
                "wind_sensitivity": self.rng.uniform(0.8, 1.2),  # Response to wind
            }
    
    def t_s(self) -> float:
        """Get elapsed time in seconds (scaled)"""
        return (time.time() - self.t0) * self.time_scale
    
    def now_ms(self) -> int:
        """Get current wall-clock time in milliseconds"""
        return int(time.time() * 1000)
    
    def get_environmental_factor(self, t: float) -> float:
        """
        Calculate environmental strain factor based on scenario timeline.
        
        Returns:
            Additional strain in μE due to environmental conditions
        """
        cfg = SCENARIO_CONFIG
        
        # Phase 1: Baseline (before wind starts)
        if t < cfg["wind_start_time"]:
            return 0.0
        
        # Phase 2: Wind building up
        elif t < cfg["wind_start_time"] + cfg["wind_buildup_duration"]:
            progress = (t - cfg["wind_start_time"]) / cfg["wind_buildup_duration"]
            # Smooth S-curve for realistic buildup
            smooth_progress = 3 * progress**2 - 2 * progress**3
            return smooth_progress * cfg["wind_max_strain"]
        
        # Phase 3: Check for sudden gust
        elif cfg["gust_time"] <= t < cfg["gust_time"] + cfg["gust_duration"]:
            # Sudden gust with quick rise and gradual fall
            dt = t - cfg["gust_time"]
            gust_factor = math.exp(-dt / (cfg["gust_duration"] / 3)) * \
                         (1.0 - math.exp(-10 * dt / cfg["gust_duration"]))
            return cfg["wind_max_strain"] + gust_factor * (cfg["gust_peak_strain"] - cfg["wind_max_strain"])
        
        # Phase 4: Steady wind (between buildup and subsiding)
        elif t < cfg["wind_end_time"]:
            return cfg["wind_max_strain"]
        
        # Phase 5: Wind subsiding
        elif t < cfg["wind_end_time"] + cfg["wind_subside_duration"]:
            progress = (t - cfg["wind_end_time"]) / cfg["wind_subside_duration"]
            # Smooth decay back to baseline
            smooth_progress = 1.0 - (3 * progress**2 - 2 * progress**3)
            return smooth_progress * cfg["wind_max_strain"]
        
        # Phase 6: Back to baseline
        else:
            return 0.0
    
    def generate_strain_readings(self, t: float) -> Dict[str, Dict]:
        """
        Generate realistic strain readings for all sensors.
        
        Args:
            t: Elapsed time in seconds
            
        Returns:
            Dictionary with sensor readings
        """
        self.seq += 1
        
        cfg = SCENARIO_CONFIG
        env_factor = self.get_environmental_factor(t)
        
        readings = {}
        
        for sensor_id in SENSOR_IDS:
            profile = self.sensor_profiles[sensor_id]
            pos = SENSOR_POSITIONS[sensor_id]
            phase = SENSOR_PHASE[sensor_id]
            
            # 1. Baseline strain (normal condition)
            baseline = cfg["baseline_strain"] + profile["baseline_offset"]
            
            # 2. Natural variation (slight random drift)
            variation = self.rng.gauss(0, cfg["baseline_variation"] / 3)
            
            # 3. Small structural vibration (always present)
            vibration = 0.0
            for mode in VIBRATION_MODES:
                vibration += mode["amp_uE"] * math.sin(2 * math.pi * mode["f_hz"] * t + phase)
            
            # 4. Environmental loading (wind, gusts)
            environmental = env_factor * profile["wind_sensitivity"]
            
            # 5. Sensor noise
            noise = self.rng.gauss(0, profile["noise_level"])
            
            # Total strain
            strain_uE = baseline + variation + vibration + environmental + noise
            
            # Clip to physically reasonable range
            strain_uE = max(-1500.0, min(1500.0, strain_uE))
            
            readings[sensor_id] = {
                "strain_uE": round(strain_uE, 2),
                "grid_i": pos["grid_i"],
                "grid_j": pos["grid_j"],
                "grid_k": pos["grid_k"],
                "timestamp_ms": self.now_ms(),
                "sequence": self.seq,
                "scenario_time_s": round(t, 1),
                "env_factor_uE": round(environmental, 2)
            }
        
        return readings
    
    def get_fem_input(self, t: float) -> List[float]:
        """
        Get sensor readings in format required for FEM simulation.
        
        Returns:
            List of 4 strain_uE values [S1, S2, S3, S4]
        """
        readings = self.generate_strain_readings(t)
        return [readings[sensor_id]["strain_uE"] for sensor_id in SENSOR_IDS]
    
    def get_scenario_status(self, t: float) -> str:
        """Get human-readable status of current scenario phase"""
        cfg = SCENARIO_CONFIG
        
        if t < cfg["wind_start_time"]:
            return "CALM - Normal baseline conditions"
        elif t < cfg["wind_start_time"] + cfg["wind_buildup_duration"]:
            return "WIND BUILDING - Gradual increase"
        elif cfg["gust_time"] <= t < cfg["gust_time"] + cfg["gust_duration"]:
            return "⚠️  WIND GUST - Sudden spike!"
        elif t < cfg["wind_end_time"]:
            return "WINDY - Steady elevated strain"
        elif t < cfg["wind_end_time"] + cfg["wind_subside_duration"]:
            return "WIND SUBSIDING - Returning to normal"
        else:
            return "CALM - Back to baseline"


# ================== MQTT Publishing ==================
def _make_client(client_id: str):
    """Create MQTT client"""
    if mqtt is None:
        return None
    if hasattr(mqtt, "CallbackAPIVersion"):
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id, protocol=mqtt.MQTTv311)
    return mqtt.Client(client_id=client_id, clean_session=True, protocol=mqtt.MQTTv311)


def connect_mqtt(args):
    """Connect to MQTT broker"""
    if mqtt is None:
        print("paho-mqtt is not installed. Install with: pip install paho-mqtt", file=sys.stderr)
        sys.exit(1)
    
    cid = args.client_id or f"sensor-gen-{random.randint(1000,9999)}"
    client = _make_client(cid)
    
    if args.username or args.password:
        client.username_pw_set(args.username or "", args.password or "")
    
    client.will_set(
        f"{args.topic_root}/status",
        payload=json.dumps({"state": "offline"}),
        qos=0,
        retain=True
    )
    
    client.connect(args.broker, args.port, keepalive=30)
    client.publish(
        f"{args.topic_root}/status",
        json.dumps({"state": "online"}),
        qos=0,
        retain=True
    )
    
    return client


def publish_json(client, topic: str, payload: dict, retain=False, qos=0):
    """Publish JSON payload to MQTT topic"""
    if client:
        client.publish(topic, json.dumps(payload), qos=qos, retain=retain)


def run(args):
    """Run realistic sensor generator"""
    client = connect_mqtt(args) if args.broker else None
    gen = RealisticSensorGenerator(seed=args.seed, time_scale=args.time_scale)
    
    period = max(0.05, float(args.period_s))
    next_tick = time.time()
    
    print("="*70)
    print("REALISTIC SENSOR GENERATOR - Environmental Scenario")
    print("="*70)
    print(f"\nFixed sensor positions:")
    for sensor_id, pos in SENSOR_POSITIONS.items():
        print(f"  {sensor_id}: grid_i={pos['grid_i']}, grid_j={pos['grid_j']}, grid_k={pos['grid_k']}")
    
    print(f"\nScenario timeline (time_scale={args.time_scale}x):")
    cfg = SCENARIO_CONFIG
    print(f"  0:00 - {cfg['wind_start_time']/60:.0f}m: Calm baseline (~{cfg['baseline_strain']:.0f} μE)")
    print(f"  {cfg['wind_start_time']/60:.0f}m - {(cfg['wind_start_time']+cfg['wind_buildup_duration'])/60:.0f}m: Wind building up to ~{cfg['wind_max_strain']:.0f} μE")
    print(f"  {cfg['gust_time']/60:.0f}m: ⚠️  SUDDEN GUST (spike to ~{cfg['gust_peak_strain']:.0f} μE)")
    print(f"  {cfg['wind_end_time']/60:.0f}m - {(cfg['wind_end_time']+cfg['wind_subside_duration'])/60:.0f}m: Wind subsiding")
    print(f"  After {(cfg['wind_end_time']+cfg['wind_subside_duration'])/60:.0f}m: Back to calm")
    
    print(f"\nUpdate interval: {period}s")
    print("Press Ctrl+C to stop\n")
    print("="*70)
    
    try:
        while True:
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue
            
            t_s = gen.t_s()
            
            # Generate strain readings
            readings = gen.generate_strain_readings(t_s)
            fem_input = gen.get_fem_input(t_s)
            status = gen.get_scenario_status(t_s)
            
            # Print to console
            time_str = time.strftime('%H:%M:%S')
            scenario_time = f"{int(t_s//60)}:{int(t_s%60):02d}"
            env_factor = readings["S1"]["env_factor_uE"]
            
            print(f"[{time_str}] t={scenario_time} | {status:40s} | Sensors: {fem_input} μE (env: +{env_factor:.1f})")
            
            # Publish to MQTT if broker specified
            if client:
                payload = {
                    "seq": gen.seq,
                    "t_ms": gen.now_ms(),
                    "scenario_time_s": t_s,
                    "scenario_status": status,
                    "sensors": readings,
                    "fem_input": fem_input,
                    "environmental_factor": env_factor
                }
                publish_json(client, f"{args.topic_root}/strain", payload)
            
            next_tick += period
            
    except KeyboardInterrupt:
        print("\n\nStopping sensor generator...")
    finally:
        if client:
            try:
                publish_json(client, f"{args.topic_root}/status", {"state": "offline"}, retain=True)
                client.disconnect()
            except Exception:
                pass


def build_argparser():
    """Build argument parser"""
    p = argparse.ArgumentParser(
        description="Realistic sensor generator with environmental scenarios"
    )
    p.add_argument("--broker", default="", help="MQTT broker (optional, omit for console-only)")
    p.add_argument("--port", type=int, default=1883, help="MQTT port (default: 1883)")
    p.add_argument("--topic-root", default="dome", help="Root topic (default: dome)")
    p.add_argument("--period-s", type=float, default=DEFAULT_PERIOD_S,
                   help=f"Seconds between updates (default: {DEFAULT_PERIOD_S})")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="Time scaling factor (e.g., 0.1 = 10x faster, default: 1.0)")
    p.add_argument("--client-id", default="", help="MQTT client id")
    p.add_argument("--username", default="", help="MQTT username")
    p.add_argument("--password", default="", help="MQTT password")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)

