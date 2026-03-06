#!/usr/bin/env python3
"""
Sensor Update Tool for Neo4j
Updates sensor readings directly to Neo4j at fixed voxel positions.

Sensors are fixed at specific grid positions:
- S1: (20, 15, 40)
- S2: (50, 15, 40)
- S3: (35, 25, 30)
- S4: (65, 10, 50)
"""

from neo4j import GraphDatabase
from datetime import datetime
from typing import List, Dict, Any, Optional
try:
    from ..config.settings import Settings
except ImportError:
    from config.settings import Settings


# Fixed sensor positions (grid_i, grid_j, grid_k)
SENSOR_POSITIONS = {
    "S1": (20, 15, 40),
    "S2": (50, 15, 40),
    "S3": (35, 25, 30),
    "S4": (65, 10, 50),
}


def update_sensor_readings(sensor_readings: List[float], timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Update sensor readings in Neo4j at fixed voxel positions.
    Also append each reading set to a global SensorStream node.
    
    Args:
        sensor_readings: List of 4 strain_uE values [S1, S2, S3, S4] in microstrain
        
    Returns:
        Dictionary with update results
    """
    if len(sensor_readings) != 4:
        raise ValueError(f"Expected 4 sensor readings, got {len(sensor_readings)}")
    
    conn_info = Settings.get_connection_info()
    driver = GraphDatabase.driver(
        conn_info["uri"],
        auth=(conn_info["user"], conn_info["password"])
    )
    
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    sensor_ids = ["S1", "S2", "S3", "S4"]
    
    updated_count = 0
    
    with driver.session(database=conn_info.get("database")) as session:
        for sensor_id, strain_uE in zip(sensor_ids, sensor_readings):
            grid_i, grid_j, grid_k = SENSOR_POSITIONS[sensor_id]
            
            query = """
                MATCH (v:Voxel {
                    grid_i: $grid_i,
                    grid_j: $grid_j,
                    grid_k: $grid_k
                })
                SET v.sensor_strain_uE = $strain_uE,
                    v.last_updated = $timestamp
                RETURN v.grid_i as i, v.grid_j as j, v.grid_k as k
            """
            
            result = session.run(query, {
                "grid_i": grid_i,
                "grid_j": grid_j,
                "grid_k": grid_k,
                "strain_uE": float(strain_uE),
                "timestamp": timestamp
            })
            
            if result.single():
                updated_count += 1

        # Append the full sensor set to a single SensorStream node
        stream_query = """
            MERGE (s:SensorStream {id: 'global'})
            SET s.timestamps = coalesce(s.timestamps, []) + [$timestamp],
                s.S1 = coalesce(s.S1, []) + [$s1],
                s.S2 = coalesce(s.S2, []) + [$s2],
                s.S3 = coalesce(s.S3, []) + [$s3],
                s.S4 = coalesce(s.S4, []) + [$s4],
                s.count = coalesce(s.count, 0) + 1,
                s.last_updated = $timestamp
            RETURN s.count as count
        """
        session.run(stream_query, {
            "timestamp": timestamp,
            "s1": float(sensor_readings[0]),
            "s2": float(sensor_readings[1]),
            "s3": float(sensor_readings[2]),
            "s4": float(sensor_readings[3])
        })
    
    driver.close()
    
    return {
        "success": True,
        "sensors_updated": updated_count,
        "timestamp": timestamp,
        "readings": dict(zip(sensor_ids, sensor_readings))
    }


def get_sensor_readings() -> List[float]:
    """
    Retrieve current sensor readings from Neo4j.
    
    Returns:
        List of 4 strain_uE values [S1, S2, S3, S4]
    """
    conn_info = Settings.get_connection_info()
    driver = GraphDatabase.driver(
        conn_info["uri"],
        auth=(conn_info["user"], conn_info["password"])
    )
    
    sensor_ids = ["S1", "S2", "S3", "S4"]
    readings = []
    
    with driver.session(database=conn_info.get("database")) as session:
        for sensor_id in sensor_ids:
            grid_i, grid_j, grid_k = SENSOR_POSITIONS[sensor_id]
            
            query = """
                MATCH (v:Voxel {
                    grid_i: $grid_i,
                    grid_j: $grid_j,
                    grid_k: $grid_k
                })
                RETURN v.sensor_strain_uE as strain_uE
            """
            
            result = session.run(query, {
                "grid_i": grid_i,
                "grid_j": grid_j,
                "grid_k": grid_k
            })
            
            record = result.single()
            if record and record["strain_uE"] is not None:
                readings.append(record["strain_uE"])
            else:
                readings.append(None)
    
    driver.close()
    
    return readings


def reset_sensor_stream() -> None:
    """
    Reset the global SensorStream log by deleting existing nodes.
    Called on pipeline startup to start history from zero.
    """
    conn_info = Settings.get_connection_info()
    driver = GraphDatabase.driver(
        conn_info["uri"],
        auth=(conn_info["user"], conn_info["password"])
    )
    with driver.session(database=conn_info.get("database")) as session:
        session.run("MATCH (s:SensorStream) DETACH DELETE s")
    driver.close()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Update sensors
        readings = [float(x) for x in sys.argv[1:5]]
        result = update_sensor_readings(readings)
        print(f"Updated {result['sensors_updated']} sensors")
        print(f"Readings: {result['readings']}")
    else:
        # Get current readings
        readings = get_sensor_readings()
        print(f"Current sensor readings: {readings}")

