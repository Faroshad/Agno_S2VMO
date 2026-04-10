#!/usr/bin/env python3
"""
Synchronized Simulation Coordinator
Orchestrates the complete simulation cycle:
1. Reset Neo4j to initial state on startup
2. Read real sensor data from the physical dome via MQTT
3. Update Neo4j with sensor data
4. Trigger FEM simulation with new sensor data
5. Update Neo4j with FEM results

All components are synchronized with the MQTT data rate from the physical model.
"""

import sys
import os
import time
import threading
import argparse
from collections import deque
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from initialize_neo4j import reset_neo4j_to_initial_state
from agents.monitoring_agent import MonitoringAgent
from config.settings import Settings
from core.event_bus import event_bus, SimulationCycleEvent
from tools.sensor_update import update_sensor_readings, reset_sensor_stream
from tools.fem_tool import run_pipeline, update_neo4j_with_fem_results
import numpy as np

# ── MQTT defaults (match mqtt.py) ─────────────────────────────────────────────
DEFAULT_MQTT_BROKER = "broker.hivemq.com"
DEFAULT_MQTT_PORT   = 1883
DEFAULT_MQTT_TOPIC  = "farshad/s2vmo/dome/stream"


def _extract_sensor_values(payload: dict):
    """Extract [S1, S2, S3, S4] floats from a decoded MQTT payload dict."""
    if not isinstance(payload, dict):
        return None
    # Format: {"fem_input": [s1, s2, s3, s4]}
    if "fem_input" in payload and isinstance(payload["fem_input"], list) and len(payload["fem_input"]) >= 4:
        return [float(v) for v in payload["fem_input"][:4]]
    # Format: {"sensors": {"S1": {"strain_uE": x}, ...}}
    if "sensors" in payload and isinstance(payload["sensors"], dict):
        result = []
        for key in ["S1", "S2", "S3", "S4"]:
            s = payload["sensors"].get(key)
            if s is None:
                break
            val = s.get("strain_uE", s.get("value", s.get("strain", s.get("uE", 0)))) if isinstance(s, dict) else s
            result.append(float(val))
        if len(result) == 4:
            return result
    # Format: {"S1": x, "S2": x, "S3": x, "S4": x}
    result = [float(payload[k]) for k in ["S1", "S2", "S3", "S4"] if k in payload]
    if len(result) == 4:
        return result
    # Format: flat list under various keys
    for field in ("strain", "readings", "values", "data"):
        v = payload.get(field)
        if isinstance(v, list) and len(v) >= 4:
            return [float(x) for x in v[:4]]
    return None


class SimulationCoordinator:
    """
    Coordinates synchronized simulation driven by real MQTT sensor data:
    - MQTT listener (physical model)
    - Neo4j sensor updates
    - FEM analysis
    - Neo4j FEM result storage
    """

    def __init__(self, update_period: float = 3.0,
                 mqtt_broker: str = DEFAULT_MQTT_BROKER,
                 mqtt_port: int = DEFAULT_MQTT_PORT,
                 mqtt_topic: str = DEFAULT_MQTT_TOPIC):
        self.update_period = update_period
        self.cycle_count = 0
        self.start_time = None
        self.fem_skips = 0
        self.skip_next_fem = False
        self.cycle_durations = deque(maxlen=100)
        self.last_monitoring_summary = None

        # MQTT shared state
        self._latest_sensor_values = None
        self._new_data_event = threading.Event()
        self._mqtt_broker = mqtt_broker
        self._mqtt_port   = mqtt_port
        self._mqtt_topic  = mqtt_topic
        self._mqtt_client = None

        try:
            self.monitoring_agent = MonitoringAgent()
        except Exception as exc:
            self.monitoring_agent = None
            print(f"⚠️  Monitoring agent disabled: {exc}")

    # ── MQTT helpers ──────────────────────────────────────────────────────────
    def _start_mqtt(self):
        """Start MQTT subscriber in a background thread."""
        try:
            import paho.mqtt.client as mqtt_lib
        except ImportError:
            print("❌ paho-mqtt not installed. Run: pip install paho-mqtt")
            return

        def on_connect(client, userdata, flags, rc, props=None):
            if rc == 0:
                client.subscribe(self._mqtt_topic)
                print(f"📡 MQTT subscribed → {self._mqtt_topic}")
            else:
                print(f"❌ MQTT connect failed rc={rc}")

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
            except Exception:
                return
            vals = _extract_sensor_values(payload)
            if vals:
                self._latest_sensor_values = vals
                self._new_data_event.set()

        if hasattr(mqtt_lib, "CallbackAPIVersion"):
            client = mqtt_lib.Client(mqtt_lib.CallbackAPIVersion.VERSION2)
        else:
            client = mqtt_lib.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        self._mqtt_client = client

        def _run():
            try:
                client.connect(self._mqtt_broker, self._mqtt_port, 60)
                client.loop_forever()
            except Exception as e:
                print(f"❌ MQTT thread error: {e}")

        threading.Thread(target=_run, daemon=True, name="coordinator-mqtt").start()
        print(f"🔌 Connecting to MQTT broker {self._mqtt_broker}:{self._mqtt_port}…")

    def _wait_for_sensor_data(self, timeout: float = 15.0):
        """Block until new MQTT sensor data arrives (or timeout). Returns values or None."""
        self._new_data_event.clear()
        got = self._new_data_event.wait(timeout=timeout)
        if got:
            return list(self._latest_sensor_values)
        # Timeout — use last known values if available
        if self._latest_sensor_values is not None:
            return list(self._latest_sensor_values)
        return None

    def _avg_cycle_duration(self) -> float:
        if not self.cycle_durations:
            return 0.0
        return float(sum(self.cycle_durations) / len(self.cycle_durations))

    def _print_runtime_status(self):
        print("\n📈 Runtime status")
        print(f"   cycles_completed: {self.cycle_count}")
        print(f"   fem_skips: {self.fem_skips}")
        print(f"   avg_cycle_time_s: {self._avg_cycle_duration():.2f}")
        if self.last_monitoring_summary:
            print(
                "   last_alerts: "
                f"stress_breaches={self.last_monitoring_summary.get('stress_breaches', 0)}, "
                f"quality_issues={self.last_monitoring_summary.get('quality_issues', 0)}"
            )
        
    def initialize_system(self):
        """Initialize/reset the system to initial state"""
        print("\n" + "="*80)
        print("SYSTEM INITIALIZATION")
        print("="*80)
        
        print("\n📊 Step 1: Resetting Neo4j to initial state...")
        try:
            reset_neo4j_to_initial_state()
            print("✅ Neo4j reset complete")
        except Exception as e:
            print(f"❌ Error resetting Neo4j: {e}")
            raise
        
        print("\n✅ System initialization complete!")
        print("="*80)
        
        # Reset SensorStream history so it starts fresh for this run
        try:
            reset_sensor_stream()
            print("🧹 SensorStream history reset")
        except Exception as e:
            print(f"⚠️  Warning: could not reset SensorStream: {e}")
    
    def run_cycle(self, run_fem: bool = True) -> dict:
        """
        Run one complete simulation cycle driven by real MQTT sensor data:
        1. Read sensor data from MQTT (physical model)
        2. Update Neo4j with sensor data
        3. Run FEM simulation
        4. Update Neo4j with FEM results

        Returns:
            Dict with cycle statistics
        """
        cycle_start = time.time()

        print(f"\n{'─'*80}")
        print(f"CYCLE {self.cycle_count} | {datetime.now().strftime('%H:%M:%S')} | Source: MQTT Physical Model")
        print(f"{'─'*80}")

        # Step 1: Get real sensor readings from MQTT
        sensor_values = self._wait_for_sensor_data(timeout=self.update_period * 2)
        if sensor_values is None:
            print("⚠️  No MQTT sensor data received — skipping cycle")
            return {"success": False, "error": "No MQTT sensor data"}
        status = "MQTT · Physical Model Live"

        print(f"📊 Sensor Source: {status}")
        print(f"📊 Sensor Readings: {[f'{v:.1f}' for v in sensor_values]} μE")
        
        # Step 2: Update Neo4j with sensor data
        print(f"\n🔄 Updating Neo4j with sensor readings...")
        try:
            # Use a single cycle timestamp for all writes in this cycle
            cycle_timestamp = datetime.now().isoformat()
            sensor_update_result = update_sensor_readings(sensor_values, timestamp=cycle_timestamp)
            print(f"✅ Updated {sensor_update_result['sensors_updated']} sensor voxels in Neo4j")
        except Exception as e:
            print(f"❌ Error updating sensors in Neo4j: {e}")
            return {"success": False, "error": str(e)}
        
        fem_results = None
        timestamp = cycle_timestamp

        # Step 3: Run FEM simulation
        if run_fem:
            print(f"\n🔬 Running FEM simulation...")
            try:
                # Convert to numpy array for FEM pipeline
                sensor_array = np.array(sensor_values).reshape(1, -1)

                # Run FEM pipeline
                fem_results = run_pipeline(sensor_reading=sensor_array, output_dir="out")

                print(f"✅ FEM simulation complete")
                print(f"   Avg stress: {fem_results['stress_magnitude'].mean():.2e} Pa")
                print(f"   Max stress: {fem_results['stress_magnitude'].max():.2e} Pa")

            except Exception as e:
                print(f"❌ Error running FEM simulation: {e}")
                return {"success": False, "error": str(e)}

            # Step 4: Update Neo4j with FEM results
            print(f"\n📊 Updating Neo4j with FEM results...")
            try:
                # Reuse the same cycle timestamp to keep sensor/FEM writes in sync
                neo4j_result = update_neo4j_with_fem_results(
                    strain_voxel=fem_results['strain_voxel'],
                    stress_voxel=fem_results['stress_voxel'],
                    voxel_mask=fem_results['voxel_mask'],
                    solid_coords=fem_results['solid_coords'],
                    sensor_reading=np.array(sensor_values),
                    timestamp=timestamp
                )

                print(f"✅ Updated {neo4j_result['voxels_updated']} voxels with FEM results")

            except Exception as e:
                print(f"❌ Error updating Neo4j with FEM results: {e}")
                return {"success": False, "error": str(e)}
        else:
            print("\n⏭️  Skipping FEM for this cycle to recover from prior overrun")
            neo4j_result = {"voxels_updated": 0}

        # Step 5: Proactive monitoring alerts
        if self.monitoring_agent is not None:
            try:
                monitoring_summary = self.monitoring_agent.assess_cycle(
                    cycle=self.cycle_count,
                    timestamp=timestamp,
                )
                self.last_monitoring_summary = monitoring_summary
                print(
                    "✅ Monitoring alerts processed: "
                    f"stress_breaches={monitoring_summary['stress_breaches']}, "
                    f"quality_issues={monitoring_summary['quality_issues']}"
                )
            except Exception as e:
                monitoring_summary = {"error": str(e)}
                print(f"⚠️  Monitoring agent failed for cycle {self.cycle_count}: {e}")
        else:
            monitoring_summary = {"status": "disabled"}
        
        # Calculate cycle statistics
        cycle_duration = time.time() - cycle_start
        
        stats = {
            "success": True,
            "cycle": self.cycle_count,
            "status": status,
            "sensor_readings": sensor_values,
            "avg_stress": float(fem_results['stress_magnitude'].mean()) if fem_results is not None else 0.0,
            "max_stress": float(fem_results['stress_magnitude'].max()) if fem_results is not None else 0.0,
            "voxels_updated": neo4j_result['voxels_updated'],
            "cycle_duration_s": cycle_duration,
            "timestamp": timestamp,
            "fem_skipped": not run_fem,
            "monitoring": monitoring_summary,
        }

        event_bus.publish_cycle_complete(
            SimulationCycleEvent(
                cycle=self.cycle_count,
                timestamp=timestamp,
                voxels_updated=neo4j_result["voxels_updated"],
                max_stress=stats["max_stress"],
                avg_stress=stats["avg_stress"],
                fem_skipped=not run_fem,
            )
        )
        
        print(f"\n✅ Cycle {self.cycle_count} complete in {cycle_duration:.1f}s")
        print(f"{'─'*80}")
        
        return stats
    
    def run_continuous(self, max_cycles: int = None):
        """
        Run continuous synchronized simulation.
        
        Args:
            max_cycles: Maximum number of cycles to run (None = infinite)
        """
        print("\n" + "="*80)
        print("SYNCHRONIZED SIMULATION COORDINATOR")
        print("Data source: MQTT Physical Model")
        print("="*80)
        print(f"Update period: {self.update_period}s")
        print(f"MQTT broker:  {self._mqtt_broker}:{self._mqtt_port}")
        print(f"MQTT topic:   {self._mqtt_topic}")
        print(f"Max cycles:   {max_cycles if max_cycles else 'Infinite'}")
        print("="*80)

        # Start MQTT listener (background thread)
        self._start_mqtt()
        time.sleep(1.0)   # brief pause for connection

        # Initialize system
        self.initialize_system()
        
        # Start simulation
        self.start_time = time.time()
        next_cycle_time = self.start_time
        
        print("\n🚀 Starting synchronized simulation...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Check if we've reached max cycles
                if max_cycles and self.cycle_count >= max_cycles:
                    print(f"\n✅ Reached maximum cycles ({max_cycles}). Stopping.")
                    break
                
                # Wait until next cycle time
                now = time.time()
                if now < next_cycle_time:
                    time.sleep(next_cycle_time - now)
                
                # Run cycle
                try:
                    run_fem_this_cycle = not self.skip_next_fem
                    stats = self.run_cycle(run_fem=run_fem_this_cycle)
                    
                    if not stats.get("success", False):
                        print(f"❌ Cycle {self.cycle_count} failed: {stats.get('error')}")
                    else:
                        self.cycle_durations.append(stats["cycle_duration_s"])

                        if not run_fem_this_cycle:
                            self.fem_skips += 1
                            self.skip_next_fem = False
                        else:
                            overrun_limit = self.update_period * Settings.MAX_CYCLE_OVERRUN_FACTOR
                            if stats["cycle_duration_s"] > overrun_limit:
                                print(
                                    "⚠️  Cycle duration exceeded supervision threshold; "
                                    "next cycle will skip FEM for recovery"
                                )
                                self.skip_next_fem = True

                        if self.cycle_count > 0 and self.cycle_count % 5 == 0:
                            self._print_runtime_status()
                    
                except Exception as e:
                    print(f"❌ Unexpected error in cycle {self.cycle_count}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Increment cycle counter
                self.cycle_count += 1
                
                # Schedule next cycle
                next_cycle_time += self.update_period
                
                # If we're falling behind, catch up
                if time.time() > next_cycle_time:
                    print(f"⚠️  Warning: Cycle took longer than update period, catching up...")
                    next_cycle_time = time.time()
        
        except KeyboardInterrupt:
            print("\n\n👋 Simulation stopped by user")
        
        finally:
            # Print summary
            total_time = time.time() - self.start_time
            print("\n" + "="*80)
            print("SIMULATION SUMMARY")
            print("="*80)
            print(f"Total cycles: {self.cycle_count}")
            print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
            print(f"Average cycle time: {total_time/max(self.cycle_count, 1):.1f}s")
            print(f"FEM skips: {self.fem_skips}")
            print("="*80)

            try:
                self.monitoring_agent.close()
            except Exception:
                pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Synchronized Simulation Coordinator — driven by real MQTT sensor data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (reads from HiveMQ broker)
  python synchronized_sim_coordinator.py

  # Run with 5s update period
  python synchronized_sim_coordinator.py --period 5

  # Run for only 10 cycles
  python synchronized_sim_coordinator.py --max-cycles 10

  # Use a custom MQTT broker
  python synchronized_sim_coordinator.py --broker 192.168.1.100 --port 1883
"""
    )

    parser.add_argument("--period", type=float, default=3.0,
                        help="Max wait period per cycle in seconds (default: 3.0)")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="Maximum number of cycles to run (default: infinite)")
    parser.add_argument("--broker", default=DEFAULT_MQTT_BROKER,
                        help=f"MQTT broker hostname (default: {DEFAULT_MQTT_BROKER})")
    parser.add_argument("--port", type=int, default=DEFAULT_MQTT_PORT,
                        help=f"MQTT broker port (default: {DEFAULT_MQTT_PORT})")
    parser.add_argument("--topic", default=DEFAULT_MQTT_TOPIC,
                        help=f"MQTT topic (default: {DEFAULT_MQTT_TOPIC})")

    args = parser.parse_args()

    coordinator = SimulationCoordinator(
        update_period=args.period,
        mqtt_broker=args.broker,
        mqtt_port=args.port,
        mqtt_topic=args.topic,
    )
    
    try:
        coordinator.run_continuous(max_cycles=args.max_cycles)
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

