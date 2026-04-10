#!/usr/bin/env python3
"""
Initialize Neo4j Graph Database from Voxel Grid

This script:
1. Runs voxelizer to create voxel_grid.npz
2. Clears existing Neo4j database
3. Creates new Neo4j graph with all voxels (stress/strain initialized as NaN)
"""

import sys
import os
import io
import subprocess

# Force UTF-8 stdout so emoji / special chars never crash on Windows CP1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add parent directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.voxel_grid_to_neo4j import VoxelGridToNeo4j


def reset_neo4j_to_initial_state():
    """
    Reset Neo4j database to initial state with only voxel structure.
    All stress/strain values are reset to NaN (None in Neo4j).
    """
    print("\n🔄 Resetting Neo4j to initial state...")
    builder = VoxelGridToNeo4j()
    try:
        builder.build_graph_from_voxel_grid(
            voxel_grid_path="out/voxel_grid.npz",
            clear_db=True
        )
        print("✅ Neo4j reset to initial state (voxel structure only, no FEM data)")
    except Exception as e:
        print(f"❌ Error resetting Neo4j: {e}")
        raise
    finally:
        builder.close()


def main():
    print("="*70)
    print("NEO4J GRAPH DATABASE INITIALIZATION")
    print("="*70)
    
    # Step 1: Run voxelizer
    print("\n📦 Step 1: Creating voxel grid from mesh...")
    try:
        voxelizer_path = os.path.join("utils", "voxelizer.py")
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, voxelizer_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running voxelizer: {e}")
        print(e.stdout)
        print(e.stderr)
        return 1
    
    # Step 2: Initialize Neo4j from voxel grid
    print("\n📊 Step 2: Building Neo4j graph database...")
    reset_neo4j_to_initial_state()
    
    print("\n" + "="*70)
    print("✅ INITIALIZATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Run FEM analysis to populate stress/strain values")
    print("2. Update sensors to populate sensor readings")
    print("3. Query Neo4j to retrieve updated voxel data")
    
    return 0


if __name__ == "__main__":
    exit(main())


# Export for use in other modules
__all__ = ["main", "reset_neo4j_to_initial_state"]

