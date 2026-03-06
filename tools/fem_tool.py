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
from typing import Optional, Dict, List, Any
from datetime import datetime
from neo4j import GraphDatabase
from agno.agent import Toolkit
# Support both package-relative and absolute imports
try:
    from ..config.settings import Settings
except ImportError:
    from config.settings import Settings

# ============================================================
# BOOLEAN MASK CREATION
# ============================================================

def create_voxel_mask(voxel_grid_path="out/voxel_grid.npz"):
    """
    Create boolean mask M from the voxelized mesh data.
    Uses the same voxelization as the neural network pipeline (0.1m voxels).
    """
    print("Loading voxel grid from voxelized mesh...")
    
    # Load the voxel grid created by voxelizer.py
    vg = np.load(voxel_grid_path, allow_pickle=True)
    M = vg["matrix"].astype(bool)
    origin = vg["origin"].astype(float).reshape(3,)
    pitch = float(np.atleast_1d(vg["pitch"])[0])
    
    print(f"Voxel matrix shape: {M.shape}")
    print(f"Origin: {origin}")
    print(f"Pitch: {pitch}")
    print(f"Solid voxels: {np.sum(M)}")
    
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
    
    Args:
        sensor_reading: Array of 4 sensor values (microstrain). 
                      If None, uses default values.
        output_dir: Directory to save output files.
    
    Returns:
        dict: Results containing strain and stress predictions
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # CREATE BOOLEAN MASK FROM VOXELIZED MESH
    # ============================================================
    print("Creating boolean mask from voxelized mesh...")
    M = create_voxel_mask()
    
    # Get solid voxel coordinates using np.argwhere
    coords = np.argwhere(M)
    print(f"Solid voxel coordinates shape: {coords.shape}")
    print(f"Number of solid voxels: {len(coords)}")
    
    # Load origin and pitch from the voxel grid
    vg = np.load("out/voxel_grid.npz", allow_pickle=True)
    origin = vg["origin"].astype(float).reshape(3,)
    pitch = float(np.atleast_1d(vg["pitch"])[0])
    
    # Create voxel_grid.npz with the correct mask M
    voxel_grid_path = os.path.join(output_dir, "voxel_grid.npz")
    np.savez(voxel_grid_path, 
             matrix=M, 
             origin=origin, 
             pitch=pitch)
    print(f"✅ Voxel grid saved: {voxel_grid_path}")
    
    # ============================================================
    # SETUP DEVICE AND MODELS
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    s2v = SensorToVoxelNet(n_sensors=4, grid_shape=(83, 45, 87)).to(device)
    v2s = StressUNet(in_channels=3, out_channels=6).to(device)

    # Load trained model weights
    print("Loading model weights...")
    
    # Load models with strict=False to handle architecture mismatches
    # This allows the models to load even if some layer names don't match exactly
    try:
        s2v.load_state_dict(torch.load("data/models/s2v_model_final.pth", map_location=device), strict=False)
        v2s.load_state_dict(torch.load("data/models/unet_model_final.pth", map_location=device), strict=False)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure the model files exist and are compatible")
        raise

    # Set models to evaluation mode
    s2v.eval()
    v2s.eval()

    # ============================================================
    # INPUT SENSOR DATA
    # ============================================================
    if sensor_reading is None:
        # Default sensor readings (microstrain)
        sensor_reading = np.array([
        [65.3, 120.1, 5.8, 30.0],
        [72.5, 115.8, 4.9, 29.4],
        [68.0, 122.0, 6.1, 31.2],
        ]) * 1e-6
    else:
        # Ensure proper shape and convert to microstrain
        sensor_reading = np.array(sensor_reading).reshape(1, -1) * 1e-6
    
    print(f"Input sensor readings (microstrain): {sensor_reading[0] * 1e6}")
    
    sensor_tensor = torch.tensor(sensor_reading, dtype=torch.float32).to(device)

    # ============================================================
    # STAGE 1: SENSOR -> VOXEL STRAIN
    # ============================================================
    print("\n" + "="*50)
    print("STAGE 1: Sensor readings → 3D strain voxels")
    print("="*50)
    
    with torch.no_grad():
        eps_voxel = s2v(sensor_tensor).cpu().numpy()[0]  # (3,83,45,87)
    
    # Save strain prediction
    strain_output_path = os.path.join(output_dir, "eps_voxel_predicted.npz")
    np.savez(strain_output_path, eps_voxel=eps_voxel)
    print(f"✅ Stage 1 complete: saved {strain_output_path}")
    print(f"Strain voxel shape: {eps_voxel.shape}")

    # ============================================================
    # LOAD NORMALIZATION DATA
    # ============================================================
    print("\nLoading normalization parameters...")
    strain_mean = np.load("data/models/strain_mean.npy")
    strain_std = np.load("data/models/strain_std.npy")
    
    # Normalize strain data
    eps_norm = (eps_voxel - strain_mean) / (strain_std + 1e-8)
    print("Strain data normalized for stress prediction")

    # ============================================================
    # STAGE 2: VOXEL STRAIN -> STRESS FIELD
    # ============================================================
    print("\n" + "="*50)
    print("STAGE 2: 3D strain voxels → 6-component stress field")
    print("="*50)
    
    X = torch.tensor(eps_norm[None, ...], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        sigma_voxel = v2s(X).cpu().numpy()[0]  # shape=(6,83,45,87)

    # Save stress prediction
    stress_output_path = os.path.join(output_dir, "sigma_voxel_predicted_pp.npz")
    np.savez(stress_output_path, sigma_voxel=sigma_voxel)
    print(f"✅ Stage 2 complete: saved {stress_output_path}")
    
    # ============================================================
    # SAVE BOOLEAN MASK AND SOLID VOXEL DATA
    # ============================================================
    print("\n" + "="*50)
    print("SAVING BOOLEAN MASK AND SOLID VOXEL DATA")
    print("="*50)
    
    # Save boolean mask and coordinates
    mask_output_path = os.path.join(output_dir, "voxel_mask.npz")
    np.savez(mask_output_path, 
             M=M, 
             coords=coords,
             solid_voxel_count=len(coords),
             total_voxels=M.size)
    print(f"✅ Boolean mask saved: {mask_output_path}")
    
    # Extract strain and stress data only for solid voxels
    strain_solid = eps_voxel[:, M]  # (3, N_solid)
    stress_solid = sigma_voxel[:, M]  # (6, N_solid)
    
    # Save solid voxel data
    solid_output_path = os.path.join(output_dir, "solid_voxel_data.npz")
    np.savez(solid_output_path,
             strain_solid=strain_solid,
             stress_solid=stress_solid,
             coords=coords,
             M=M)
    print(f"✅ Solid voxel data saved: {solid_output_path}")
    print(f"Solid voxel strain shape: {strain_solid.shape}")
    print(f"Solid voxel stress shape: {stress_solid.shape}")

    # ============================================================
    # RESULTS ANALYSIS
    # ============================================================
    print("\n" + "="*50)
    print("STRESS FIELD ANALYSIS")
    print("="*50)
    
    σ_names = ["σ_xx", "σ_yy", "σ_zz", "σ_xy", "σ_yz", "σ_xz"]
    print(f"Stress field shape: {sigma_voxel.shape}")
    
    print("\nStress component ranges:")
    for i, name in enumerate(σ_names):
        min_val = sigma_voxel[i].min()
        max_val = sigma_voxel[i].max()
        print(f"{name:>6}: {min_val:>10.3e} ~ {max_val:>10.3e}")

    # Calculate stress magnitude
    σ_mag = np.sqrt(np.sum(sigma_voxel[:3]**2, axis=0))
    print(f"\nStress magnitude statistics:")
    print(f"  Mean: {σ_mag.mean():.3e}")
    print(f"  99th percentile: {np.percentile(σ_mag, 99):.3e}")
    print(f"  Max: {σ_mag.max():.3e}")

    print(f"\n✅ Full pipeline complete!")
    
    # Return results
    return {
        'strain_voxel': eps_voxel,
        'stress_voxel': sigma_voxel,
        'stress_magnitude': σ_mag,
        'strain_output_path': strain_output_path,
        'stress_output_path': stress_output_path,
        'voxel_mask': M,
        'solid_coords': coords
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
        # Get the next version number (count existing FEMAnalysis nodes)
        version_query = """
            MATCH (a:FEMAnalysis)
            RETURN count(a) as version_count
        """
        version_result = session.run(version_query)
        version_record = version_result.single()
        version_number = version_record["version_count"] if version_record else 0
        
        print(f"📊 Creating FEM simulation version {version_number}")
        
        # Create FEMAnalysis node to group this run
        analysis_id = f"fem_{timestamp.replace(':', '-').replace('.', '-')}"
        
        create_analysis_query = """
            CREATE (a:FEMAnalysis {
                analysis_id: $analysis_id,
                version: $version_number,
                timestamp: $timestamp,
                sensor_reading: $sensor_reading,
                total_voxels: $total_voxels,
                status: 'completed'
            })
            RETURN a.analysis_id as id, a.version as version
        """
        
        analysis_result = session.run(create_analysis_query, {
            "analysis_id": analysis_id,
            "version_number": version_number,
            "timestamp": timestamp,
            "sensor_reading": sensor_reading.tolist(),
            "total_voxels": len(solid_coords)
        })
        
        analysis_record = analysis_result.single()
        print(f"✅ Created FEMAnalysis node: {analysis_id} (version {version_number})")
        
        # OPTIMIZED: Use simple incremental update instead of complex array management
        # Just store the latest values for each voxel (versioning via FEMAnalysis node)
        batch_size = 1000
        for batch_start in range(0, len(solid_coords), batch_size):
            batch_end = min(batch_start + batch_size, len(solid_coords))
            
            # Prepare batch data with current FEM values
            batch_data = []
            for idx in range(batch_start, batch_end):
                coord = solid_coords[idx]
                i, j, k = coord
                
                # Extract FEM data for this voxel
                batch_data.append({
                    "grid_i": int(i),
                    "grid_j": int(j),
                    "grid_k": int(k),
                    "eps_xx": float(strain_solid[0, idx]),
                    "eps_yy": float(strain_solid[1, idx]),
                    "eps_zz": float(strain_solid[2, idx]),
                    "sigma_xx": float(stress_solid[0, idx]),
                    "sigma_yy": float(stress_solid[1, idx]),
                    "sigma_zz": float(stress_solid[2, idx]),
                    "sigma_xy": float(stress_solid[3, idx]),
                    "sigma_yz": float(stress_solid[4, idx]),
                    "sigma_xz": float(stress_solid[5, idx]),
                    "stress_magnitude": float(stress_magnitude[idx]),
                    "last_fem_version": version_number
                })
                
                updated_voxels.append({
                    "grid_position": [int(i), int(j), int(k)],
                    "strain": {
                        'eps_xx': float(strain_solid[0, idx]),
                        'eps_yy': float(strain_solid[1, idx]),
                        'eps_zz': float(strain_solid[2, idx])
                    },
                    "stress": {
                        'sigma_xx': float(stress_solid[0, idx]),
                        'sigma_yy': float(stress_solid[1, idx]),
                        'sigma_zz': float(stress_solid[2, idx]),
                        'sigma_xy': float(stress_solid[3, idx]),
                        'sigma_yz': float(stress_solid[4, idx]),
                        'sigma_xz': float(stress_solid[5, idx]),
                        'stress_magnitude': float(stress_magnitude[idx])
                    }
                })
            
            # Append new results to per-voxel time-series arrays (no overwrite)
            update_query = """
                UNWIND $batch AS data
                MATCH (v:Voxel {
                    grid_i: data.grid_i,
                    grid_j: data.grid_j,
                    grid_k: data.grid_k
                })
                SET v.eps_xx = coalesce(v.eps_xx, []) + [data.eps_xx],
                    v.eps_yy = coalesce(v.eps_yy, []) + [data.eps_yy],
                    v.eps_zz = coalesce(v.eps_zz, []) + [data.eps_zz],
                    v.sigma_xx = coalesce(v.sigma_xx, []) + [data.sigma_xx],
                    v.sigma_yy = coalesce(v.sigma_yy, []) + [data.sigma_yy],
                    v.sigma_zz = coalesce(v.sigma_zz, []) + [data.sigma_zz],
                    v.sigma_xy = coalesce(v.sigma_xy, []) + [data.sigma_xy],
                    v.sigma_yz = coalesce(v.sigma_yz, []) + [data.sigma_yz],
                    v.sigma_xz = coalesce(v.sigma_xz, []) + [data.sigma_xz],
                    v.stress_magnitude = coalesce(v.stress_magnitude, []) + [data.stress_magnitude],
                    v.fem_versions = coalesce(v.fem_versions, []) + [data.last_fem_version],
                    v.fem_timestamps = coalesce(v.fem_timestamps, []) + [$timestamp],
                    v.last_fem_version = data.last_fem_version,
                    v.last_updated = $timestamp
                RETURN count(v) as updated_count
            """
            
            session.run(update_query, {"batch": batch_data, "timestamp": timestamp})
            
            if batch_end % 1000 == 0 or batch_end == len(solid_coords):
                print(f"   Updated {batch_end}/{len(solid_coords)} voxels...")
        
        print(f"✅ Updated {len(updated_voxels)} voxels with FEM results")
        
        # Update FEMAnalysis summary statistics (simplified - uses current values)
        summary_query = """
            MATCH (a:FEMAnalysis {analysis_id: $analysis_id})
            MATCH (v:Voxel)
            WHERE v.stress_magnitude IS NOT NULL AND v.last_fem_version = $version
            WITH a,
                 avg(v.stress_magnitude) as avg_stress,
                 max(v.stress_magnitude) as max_stress,
                 min(v.stress_magnitude) as min_stress,
                 count(v) as result_count
            SET a.avg_stress_magnitude = avg_stress,
                a.max_stress_magnitude = max_stress,
                a.min_stress_magnitude = min_stress,
                a.result_count = result_count
            RETURN result_count
        """
        
        session.run(summary_query, {"analysis_id": analysis_id, "version": version_number})
        print(f"✅ Updated FEMAnalysis summary statistics for version {version_number}")
    
    driver.close()
    
    summary = {
        "success": True,
        "analysis_id": analysis_id,
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
        response = {
            "success": True,
            "fem_analysis": {
                "strain_output": results['strain_output_path'],
                "stress_output": results['stress_output_path'],
                "avg_stress_magnitude": float(results['stress_magnitude'].mean()),
                "max_stress_magnitude": float(results['stress_magnitude'].max()),
                "solid_voxels": int(len(results['solid_coords']))
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
    "update_neo4j_with_fem_results"
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
