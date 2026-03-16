#!/usr/bin/env python3
"""
Synchronized Simulation Coordinator
Orchestrates the complete simulation cycle:
1. Reset Neo4j to initial state on startup
2. Generate realistic sensor readings
3. Update Neo4j with sensor data
4. Trigger FEM simulation with new sensor data
5. Update Neo4j with FEM results

All components are synchronized with a configurable time period.
"""

import sys
import os
import time
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
from sensors.realistic_sensor_generator import RealisticSensorGenerator
from tools.sensor_update import update_sensor_readings, reset_sensor_stream
from tools.fem_tool import run_pipeline, update_neo4j_with_fem_results
import numpy as np


class SimulationCoordinator:
    """
    Coordinates synchronized simulation of:
    - Sensor data generation
    - Neo4j updates
    - FEM analysis
    """
    
    def __init__(self, update_period: float = 3.0, time_scale: float = 1.0, seed: int = 42):
        """
        Args:
            update_period: Time period between updates in seconds
            time_scale: Time scaling for scenarios (e.g., 0.1 = 10x faster)
            seed: Random seed for sensor generator
        """
        self.update_period = update_period
        self.time_scale = time_scale
        self.sensor_gen = RealisticSensorGenerator(seed=seed, time_scale=time_scale)
        self.cycle_count = 0
        self.start_time = None
        self.fem_skips = 0
        self.skip_next_fem = False
        self.cycle_durations = deque(maxlen=100)
        self.last_monitoring_summary = None

        try:
            self.monitoring_agent = MonitoringAgent()
        except Exception as exc:
            self.monitoring_agent = None
            print(f"⚠️  Monitoring agent disabled due to initialization error: {exc}")

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
        Run one complete simulation cycle:
        1. Generate sensor readings
        2. Update Neo4j with sensor data
        3. Run FEM simulation
        4. Update Neo4j with FEM results
        
        Returns:
            Dict with cycle statistics
        """
        cycle_start = time.time()
        t_scenario = self.sensor_gen.t_s()
        
        # Step 1: Generate sensor readings
        print(f"\n{'─'*80}")
        print(f"CYCLE {self.cycle_count} | Scenario time: {int(t_scenario//60)}:{int(t_scenario%60):02d}")
        print(f"{'─'*80}")
        
        sensor_readings_dict = self.sensor_gen.generate_strain_readings(t_scenario)
        sensor_values = self.sensor_gen.get_fem_input(t_scenario)
        status = self.sensor_gen.get_scenario_status(t_scenario)
        
        print(f"📊 Sensor Status: {status}")
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
            "scenario_time_s": t_scenario,
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
        print("="*80)
        print(f"Update period: {self.update_period}s")
        print(f"Time scale: {self.time_scale}x")
        print(f"Max cycles: {max_cycles if max_cycles else 'Infinite'}")
        print("="*80)
        
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
        description="Synchronized Simulation Coordinator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (3s update period, normal time scale)
  python synchronized_sim_coordinator.py
  
  # Run with 5s update period
  python synchronized_sim_coordinator.py --period 5
  
  # Run with 10x faster scenario time (for testing)
  python synchronized_sim_coordinator.py --time-scale 0.1
  
  # Run for only 10 cycles
  python synchronized_sim_coordinator.py --max-cycles 10
  
  # Run with custom seed for reproducibility
  python synchronized_sim_coordinator.py --seed 123
"""
    )
    
    parser.add_argument(
        "--period", 
        type=float, 
        default=3.0,
        help="Update period in seconds (default: 3.0)"
    )
    
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Time scaling for scenario (e.g., 0.1 = 10x faster, default: 1.0)"
    )
    
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum number of cycles to run (default: infinite)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sensor generator (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Create and run coordinator
    coordinator = SimulationCoordinator(
        update_period=args.period,
        time_scale=args.time_scale,
        seed=args.seed
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

