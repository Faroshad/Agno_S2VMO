#!/usr/bin/env python3
"""
Voxel Grid to Neo4j Graph Database Converter
Creates Neo4j graph directly from voxel_grid.npz (no JSON dependency)

Schema:
- Voxel nodes: grid_i, grid_j, grid_k, x, y, z, 
              eps_xx, eps_yy, eps_zz (strain, initially NaN)
              sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz (stress, initially NaN)
              stress_magnitude (initially NaN)
              created_at, last_updated
- ADJACENT_TO relationships between neighboring voxels
"""

import numpy as np
from neo4j import GraphDatabase
from datetime import datetime
from typing import Optional, Dict, Any
import os
try:
    from ..config.settings import Settings
except ImportError:
    from config.settings import Settings


class VoxelGridToNeo4j:
    """
    Build Neo4j graph database directly from voxel_grid.npz
    
    Features:
    - Creates voxel nodes with position, stress, strain fields
    - Initializes stress/strain as NaN
    - Creates adjacency relationships
    - Uses grid indices (i, j, k) as primary identifier
    """
    
    def __init__(self):
        """Initialize Neo4j connection"""
        conn_info = Settings.get_connection_info()
        self.driver = GraphDatabase.driver(
            conn_info["uri"],
            auth=(conn_info["user"], conn_info["password"])
        )
        self.database = conn_info.get("database")
    
    def close(self):
        """Close the database connection"""
        self.driver.close()
    
    def clear_database(self, tx):
        """Clear all Voxel nodes and relationships"""
        # Clear all Voxel nodes and related nodes
        tx.run("MATCH (v:Voxel) DETACH DELETE v")
        tx.run("MATCH (a:FEMAnalysis) DETACH DELETE a")
        tx.run("MATCH (r:FEMResult) DETACH DELETE r")
        print("✓ Database cleared")
    
    def find_neighbors(self, matrix: np.ndarray, i: int, j: int, k: int) -> list:
        """
        Find 6-face neighbors of a voxel at grid position (i, j, k)
        
        Args:
            matrix: Boolean voxel matrix
            i, j, k: Grid indices
            
        Returns:
            List of (i, j, k) tuples for neighboring solid voxels
        """
        neighbors = []
        shape = matrix.shape
        
        # Check 6 face neighbors: +x, -x, +y, -y, +z, -z
        for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            
            # Check bounds
            if (0 <= ni < shape[0] and 
                0 <= nj < shape[1] and 
                0 <= nk < shape[2] and
                matrix[ni, nj, nk]):
                neighbors.append((ni, nj, nk))
        
        return neighbors
    
    def create_voxel_nodes_batch(self, tx, voxel_data: list, origin: np.ndarray, pitch: float):
        """
        Create multiple voxel nodes in a single transaction
        
        Args:
            tx: Neo4j transaction
            voxel_data: List of dicts with keys: (i, j, k)
            origin: Origin point [x, y, z]
            pitch: Voxel size
        """
        timestamp = datetime.now().isoformat()
        
        # Prepare batch data
        batch_data = []
        for voxel in voxel_data:
            i, j, k = voxel['i'], voxel['j'], voxel['k']
            
            # Convert grid indices to world coordinates
            x = origin[0] + i * pitch
            y = origin[1] + j * pitch
            z = origin[2] + k * pitch
            
            # Check if this is a sensor location
            is_sensor = False
            # Sensor positions from simple_sensor_generator.py
            sensor_positions = [
                (20, 15, 40),  # S1
                (50, 15, 40),  # S2
                (35, 25, 30),  # S3
                (65, 10, 50),  # S4
            ]
            if (i, j, k) in sensor_positions:
                is_sensor = True
            
            batch_data.append({
                "grid_i": i,
                "grid_j": j,
                "grid_k": k,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "is_sensor": is_sensor,
                "sensor_strain_uE": None,
                # Initialize strain as NaN
                "eps_xx": None,
                "eps_yy": None,
                "eps_zz": None,
                # Initialize stress as NaN
                "sigma_xx": None,
                "sigma_yy": None,
                "sigma_zz": None,
                "sigma_xy": None,
                "sigma_yz": None,
                "sigma_xz": None,
                "stress_magnitude": None,
                "created_at": timestamp,
                "last_updated": timestamp
            })
        
        # Use UNWIND for bulk creation
        query = """
            UNWIND $batch AS voxel
            MERGE (v:Voxel {
                grid_i: voxel.grid_i,
                grid_j: voxel.grid_j,
                grid_k: voxel.grid_k
            })
            SET v.x = voxel.x,
                v.y = voxel.y,
                v.z = voxel.z,
                v.eps_xx = voxel.eps_xx,
                v.eps_yy = voxel.eps_yy,
                v.eps_zz = voxel.eps_zz,
                v.sigma_xx = voxel.sigma_xx,
                v.sigma_yy = voxel.sigma_yy,
                v.sigma_zz = voxel.sigma_zz,
                v.sigma_xy = voxel.sigma_xy,
                v.sigma_yz = voxel.sigma_yz,
                v.sigma_xz = voxel.sigma_xz,
                v.stress_magnitude = voxel.stress_magnitude,
                v.sensor_strain_uE = voxel.sensor_strain_uE,
                v.is_sensor = voxel.is_sensor,
                v.created_at = voxel.created_at,
                v.last_updated = voxel.last_updated
        """
        
        tx.run(query, {"batch": batch_data})
    
    def create_neighbor_relationships_batch(self, tx, neighbor_pairs: list):
        """
        Create ADJACENT_TO relationships between neighboring voxels
        
        Args:
            tx: Neo4j transaction
            neighbor_pairs: List of tuples ((i1, j1, k1), (i2, j2, k2))
        """
        batch_data = []
        for (i1, j1, k1), (i2, j2, k2) in neighbor_pairs:
            batch_data.append({
                "i1": i1, "j1": j1, "k1": k1,
                "i2": i2, "j2": j2, "k2": k2
            })
        
        if batch_data:
            query = """
                UNWIND $batch AS pair
                MATCH (v1:Voxel {
                    grid_i: pair.i1,
                    grid_j: pair.j1,
                    grid_k: pair.k1
                })
                MATCH (v2:Voxel {
                    grid_i: pair.i2,
                    grid_j: pair.j2,
                    grid_k: pair.k2
                })
                MERGE (v1)-[:ADJACENT_TO]-(v2)
            """
            tx.run(query, {"batch": batch_data})
    
    def build_graph_from_voxel_grid(self, voxel_grid_path: str = "out/voxel_grid.npz", 
                                     clear_db: bool = True):
        """
        Build Neo4j graph from voxel_grid.npz
        
        Args:
            voxel_grid_path: Path to voxel_grid.npz file
            clear_db: Whether to clear database before building
        """
        print(f"📁 Loading voxel grid from: {voxel_grid_path}")
        
        # Load voxel grid
        vg = np.load(voxel_grid_path, allow_pickle=True)
        M = vg["matrix"].astype(bool)
        origin = vg["origin"].astype(float).reshape(3,)
        pitch = float(np.atleast_1d(vg["pitch"])[0])
        
        print(f"Voxel matrix shape: {M.shape}")
        print(f"Origin: {origin}")
        print(f"Pitch: {pitch}")
        print(f"Solid voxels: {np.sum(M)}")
        
        # Get all solid voxel coordinates
        solid_coords = np.argwhere(M)
        total_voxels = len(solid_coords)
        
        print(f"📊 Processing {total_voxels} solid voxels...")
        
        session_database = self.database if self.database else None
        with self.driver.session(database=session_database) as session:
            # Clear database if requested
            if clear_db:
                print("🗑️  Clearing existing database...")
                session.execute_write(self.clear_database)
            
            # Create voxel nodes in batches
            print("🔨 Creating voxel nodes...")
            batch_size = 1000
            for i in range(0, total_voxels, batch_size):
                batch_coords = solid_coords[i:i + batch_size]
                voxel_data = [
                    {"i": int(coord[0]), "j": int(coord[1]), "k": int(coord[2])}
                    for coord in batch_coords
                ]
                session.execute_write(self.create_voxel_nodes_batch, voxel_data, origin, pitch)
                print(f"   Created {min(i + batch_size, total_voxels)}/{total_voxels} nodes...")
            
            # Create neighbor relationships
            print("🔗 Creating neighbor relationships...")
            neighbor_pairs = []
            for coord in solid_coords:
                i, j, k = coord
                neighbors = self.find_neighbors(M, i, j, k)
                for ni, nj, nk in neighbors:
                    # Avoid duplicate pairs (only store once)
                    if (i, j, k) < (ni, nj, nk):
                        neighbor_pairs.append(((i, j, k), (ni, nj, nk)))
            
            print(f"   Found {len(neighbor_pairs)} neighbor pairs...")
            
            # Create relationships in batches
            rel_batch_size = 5000
            for i in range(0, len(neighbor_pairs), rel_batch_size):
                batch = neighbor_pairs[i:i + rel_batch_size]
                session.execute_write(self.create_neighbor_relationships_batch, batch)
                print(f"   Created {min(i + rel_batch_size, len(neighbor_pairs))}/{len(neighbor_pairs)} relationships...")
        
        print("✅ Graph database creation complete!")
        self.print_summary()
    
    def incremental_update_voxels(self, voxel_updates: list):
        """
        Incrementally update specific voxels without rebuilding entire graph.
        Only updates the specified properties, leaving others unchanged.
        
        Args:
            voxel_updates: List of dicts with keys:
                - grid_i, grid_j, grid_k (required): Voxel position
                - Any properties to update (e.g., sensor_strain_uE, eps_xx, sigma_xx, etc.)
        """
        timestamp = datetime.now().isoformat()
        
        session_database = self.database if self.database else None
        with self.driver.session(database=session_database) as session:
            # Process updates in batches for efficiency
            batch_size = 1000
            for i in range(0, len(voxel_updates), batch_size):
                batch = voxel_updates[i:i + batch_size]
                
                # Build dynamic SET clause based on properties
                query = """
                    UNWIND $batch AS update
                    MATCH (v:Voxel {
                        grid_i: update.grid_i,
                        grid_j: update.grid_j,
                        grid_k: update.grid_k
                    })
                    SET v += update.properties,
                        v.last_updated = $timestamp
                    RETURN count(v) as updated_count
                """
                
                # Prepare batch data with separated properties
                batch_data = []
                for voxel in batch:
                    # Extract position keys
                    pos_keys = {"grid_i", "grid_j", "grid_k"}
                    properties = {k: v for k, v in voxel.items() if k not in pos_keys}
                    
                    batch_data.append({
                        "grid_i": voxel["grid_i"],
                        "grid_j": voxel["grid_j"],
                        "grid_k": voxel["grid_k"],
                        "properties": properties
                    })
                
                result = session.run(query, {"batch": batch_data, "timestamp": timestamp})
                record = result.single()
                updated = record["updated_count"] if record else 0
                
                if (i + batch_size) % 5000 == 0 or (i + batch_size) >= len(voxel_updates):
                    print(f"   Incrementally updated {min(i + batch_size, len(voxel_updates))}/{len(voxel_updates)} voxels...")
        
        print(f"✅ Incremental update complete: {len(voxel_updates)} voxels updated")
    
    def print_summary(self):
        """Print database summary"""
        session_database = self.database if self.database else None
        with self.driver.session(database=session_database) as session:
            # Count nodes
            result = session.run("MATCH (v:Voxel) RETURN count(v) as total_nodes")
            total_nodes = result.single()["total_nodes"]
            
            # Count relationships
            result = session.run("MATCH ()-[r:ADJACENT_TO]-() RETURN count(r) as total_rels")
            total_rels = result.single()["total_rels"]
            
            # Count voxels with FEM data
            result = session.run("""
                MATCH (v:Voxel)
                WHERE v.eps_xx IS NOT NULL OR v.sigma_xx IS NOT NULL
                RETURN count(v) as fem_count
            """)
            fem_count = result.single()["fem_count"]
            
            print("\n" + "="*60)
            print("📊 GRAPH DATABASE SUMMARY")
            print("="*60)
            print(f"Total Voxel Nodes: {total_nodes}")
            print(f"Total ADJACENT_TO Relationships: {total_rels}")
            print(f"Voxels with FEM Data: {fem_count}")
            print("="*60)


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Neo4j graph from voxel_grid.npz")
    parser.add_argument("--voxel-grid", default="out/voxel_grid.npz",
                       help="Path to voxel_grid.npz file")
    parser.add_argument("--no-clear", action="store_true",
                       help="Don't clear database before building")
    
    args = parser.parse_args()
    
    builder = VoxelGridToNeo4j()
    try:
        builder.build_graph_from_voxel_grid(
            voxel_grid_path=args.voxel_grid,
            clear_db=not args.no_clear
        )
    finally:
        builder.close()


if __name__ == "__main__":
    main()

