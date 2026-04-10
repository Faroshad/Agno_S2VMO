#!/usr/bin/env python3
"""
Voxelizer: Convert 3D mesh to voxel grid
"""

import sys
import io
import numpy as np
import trimesh
import os

# Force UTF-8 stdout so emoji / special chars never crash on Windows CP1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OBJ_PATH = r"models/dome.obj"
VOXEL_SIZE = 0.1  # voxel

print("Loading mesh...")
mesh_tri = trimesh.load(OBJ_PATH)
print(f"Mesh loaded: {mesh_tri.vertices.shape[0]} vertices, {mesh_tri.faces.shape[0]} faces")

print("Voxelizing mesh...")
vox = mesh_tri.voxelized(pitch=VOXEL_SIZE).fill()

voxel_matrix = vox.matrix.astype(bool)
origin = vox.transform[:3, 3].copy()
pitch = vox.pitch

print(f"Voxel matrix shape: {voxel_matrix.shape}")
print(f"Origin: {origin}")
print(f"Pitch: {pitch}")
print(f"Solid voxels: {np.sum(voxel_matrix)}")

# marching cubes extract watertight cube
mesh_mc = vox.marching_cubes

os.makedirs("out", exist_ok=True)
np.savez("out/voxel_grid.npz", matrix=voxel_matrix, origin=origin, pitch=pitch)
print("voxel_grid.npz saved:", voxel_matrix.shape)

mesh_mc.export("out/voxel_surface_mc.stl")
print("Watertight?", mesh_mc.is_watertight)

# Verify the saved data
print("\n" + "="*50)
print("VERIFYING SAVED VOXEL GRID")
print("="*50)

vg = np.load("out/voxel_grid.npz", allow_pickle=True)
M = vg["matrix"].astype(bool)
origin = vg["origin"].astype(float).reshape(3,)
pitch = float(np.atleast_1d(vg["pitch"])[0])

print(f"Loaded matrix shape: {M.shape}")
print(f"Loaded origin: {origin}")
print(f"Loaded pitch: {pitch}")
print(f"Loaded solid voxels: {np.sum(M)}")

grid = np.array(M.shape)
if grid.size != 3:
    raise ValueError(f"Voxel grid shape error")

lo = origin
hi = origin + grid * pitch

print(f"voxel world bbox: x[{lo[0]:.3f},{hi[0]:.3f}]  y[{lo[1]:.3f},{hi[1]:.3f}]  z[{lo[2]:.3f},{hi[2]:.3f}]")

# ============================================================
# LOAD AND VISUALIZE PIPELINE RESULTS
# ============================================================
print("\n" + "="*50)
print("LOADING PIPELINE RESULTS")
print("="*50)

try:
    # Load strain results from pipeline
    strain_data = np.load("out/eps_voxel_predicted.npz")
    eps_voxel = strain_data["eps_voxel"]
    print(f"✅ Strain data loaded: {eps_voxel.shape}")
    
    # Load stress results from pipeline
    stress_data = np.load("out/sigma_voxel_predicted_pp.npz")
    sigma_voxel = stress_data["sigma_voxel"]
    print(f"✅ Stress data loaded: {sigma_voxel.shape}")
    
    # Extract data only for solid voxels
    solid_coords = np.argwhere(M)
    print(f"Solid voxel coordinates: {solid_coords.shape}")
    
    # Get strain and stress for solid voxels only
    strain_solid = eps_voxel[:, M]  # (3, N_solid)
    stress_solid = sigma_voxel[:, M]  # (6, N_solid)
    
    print(f"Solid voxel strain shape: {strain_solid.shape}")
    print(f"Solid voxel stress shape: {stress_solid.shape}")
    
    # Analyze strain components
    print(f"\nStrain component ranges (solid voxels only):")
    strain_names = ["ε_xx", "ε_yy", "ε_zz"]
    for i, name in enumerate(strain_names):
        min_val = strain_solid[i].min()
        max_val = strain_solid[i].max()
        mean_val = strain_solid[i].mean()
        print(f"{name:>6}: {min_val:>10.3e} ~ {max_val:>10.3e} (mean: {mean_val:>10.3e})")
    
    # Analyze stress components
    print(f"\nStress component ranges (solid voxels only):")
    stress_names = ["σ_xx", "σ_yy", "σ_zz", "σ_xy", "σ_yz", "σ_xz"]
    for i, name in enumerate(stress_names):
        min_val = stress_solid[i].min()
        max_val = stress_solid[i].max()
        mean_val = stress_solid[i].mean()
        print(f"{name:>6}: {min_val:>10.3e} ~ {max_val:>10.3e} (mean: {mean_val:>10.3e})")
    
    # Calculate stress magnitude for solid voxels
    stress_magnitude = np.sqrt(np.sum(stress_solid[:3]**2, axis=0))
    print(f"\nStress magnitude statistics (solid voxels):")
    print(f"  Mean: {stress_magnitude.mean():.3e}")
    print(f"  Max: {stress_magnitude.max():.3e}")
    print(f"  95th percentile: {np.percentile(stress_magnitude, 95):.3e}")
    print(f"  99th percentile: {np.percentile(stress_magnitude, 99):.3e}")
    
    # Save solid voxel results
    solid_results_path = "out/solid_voxel_results.npz"
    np.savez(solid_results_path,
             strain_solid=strain_solid,
             stress_solid=stress_solid,
             stress_magnitude=stress_magnitude,
             solid_coords=solid_coords,
             voxel_matrix=M,
             origin=origin,
             pitch=pitch)
    print(f"\n✅ Solid voxel results saved: {solid_results_path}")
    
except FileNotFoundError as e:
    print(f"❌ Pipeline results not found: {e}")
    print("Please run pipeline.py first to generate strain and stress predictions")
except Exception as e:
    print(f"❌ Error loading pipeline results: {e}")
