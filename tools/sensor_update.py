#!/usr/bin/env python3
"""
Sensor Update Tool for Neo4j
Updates sensor readings at voxel positions assigned via the UI.
Physical-device data format (per MQTT message):
  [
    {"id":1,"type":"structural","hx711_raw":-15294,"strain_uE":-1500.0,
     "acc_g":{"x":0.056,"y":-0.038,"z":1.062},
     "gyro_dps":{"x":-6.47,"y":-0.65,"z":0.35}},
    {"id":2, ...}
  ]
Mapping: sensor id=1 → S1 (MPU voxel) + S3 (SG voxel)
         sensor id=2 → S2 (MPU voxel) + S4 (SG voxel)
"""

from neo4j import GraphDatabase
from datetime import datetime
from typing import List, Dict, Any, Optional
try:
    from ..config.settings import Settings
except ImportError:
    from config.settings import Settings


def update_sensor_readings(
    sensor_readings: List[float],
    sensor_positions: Dict[str, Dict],
    timestamp: Optional[str] = None,
    rich_sensors: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """
    Update sensor readings in Neo4j at user-assigned voxel positions.
    Stores the full physical measurement (strain_uE, hx711_raw, acc_g, gyro_dps)
    when rich_sensors is provided, otherwise falls back to strain_uE only.

    Args:
        sensor_readings:  [S1, S2, S3, S4] strain_uE values (for FEM)
        sensor_positions: {"S1": {"grid_i": int, "grid_j": int, "grid_k": int}, ...}
        timestamp:        ISO string (default: now)
        rich_sensors:     {"S1": {strain_uE, hx711_raw, acc_x,y,z, gyro_x,y,z}, ...}
    """
    if len(sensor_readings) != 4:
        raise ValueError(f"Expected 4 sensor readings, got {len(sensor_readings)}")
    if not sensor_positions:
        raise ValueError("sensor_positions is empty — assign sensors in the UI first")

    conn_info = Settings.get_connection_info()
    driver = GraphDatabase.driver(
        conn_info["uri"],
        auth=(conn_info["user"], conn_info["password"])
    )

    if timestamp is None:
        timestamp = datetime.now().isoformat()

    sensor_ids = ["S1", "S2", "S3", "S4"]
    updated_count = 0
    MAX_HISTORY = 200  # ring-buffer cap per voxel

    with driver.session(database=conn_info.get("database")) as session:
        for sensor_id, strain_uE in zip(sensor_ids, sensor_readings):
            pos = sensor_positions.get(sensor_id)
            if pos is None:
                continue

            # Physical sensor type (MPU = accelerometer+gyro, SG = strain gauge only)
            sensor_type: str = pos.get("sensor_type", "unknown")
            is_mpu: bool = sensor_type.upper() == "MPU"

            rich = (rich_sensors or {}).get(sensor_id)

            # Base params written to every voxel for this sensor
            base_params: Dict[str, Any] = {
                "strain_uE":   float(strain_uE),
                "timestamp":   timestamp,
                "sensor_id":   sensor_id,       # e.g. "S1"
                "sensor_type": sensor_type,     # "MPU" or "SG"
                "max_history": MAX_HISTORY,
            }

            if rich and is_mpu:
                # MPU voxels get the full IMU payload
                base_params.update({
                    "hx711_raw": int(rich.get("hx711_raw", 0)),
                    "acc_x":     float(rich.get("acc_x", 0)),
                    "acc_y":     float(rich.get("acc_y", 0)),
                    "acc_z":     float(rich.get("acc_z", 0)),
                    "gyro_x":    float(rich.get("gyro_x", 0)),
                    "gyro_y":    float(rich.get("gyro_y", 0)),
                    "gyro_z":    float(rich.get("gyro_z", 0)),
                })
                # MPU voxel: full IMU payload + history arrays
                cypher = """
                    MATCH (v:Voxel {grid_i: $grid_i, grid_j: $grid_j, grid_k: $grid_k})
                    SET v.sensor_id         = $sensor_id,
                        v.sensor_type       = $sensor_type,
                        v.sensor_strain_uE  = $strain_uE,
                        v.sensor_hx711_raw  = $hx711_raw,
                        v.sensor_acc_x      = $acc_x,
                        v.sensor_acc_y      = $acc_y,
                        v.sensor_acc_z      = $acc_z,
                        v.sensor_gyro_x     = $gyro_x,
                        v.sensor_gyro_y     = $gyro_y,
                        v.sensor_gyro_z     = $gyro_z,
                        v.last_updated      = $timestamp,
                        v.sensor_strain_history = CASE
                            WHEN v.sensor_strain_history IS NULL     THEN [$strain_uE]
                            WHEN size(v.sensor_strain_history) >= $max_history
                                THEN v.sensor_strain_history[1..] + [$strain_uE]
                            ELSE v.sensor_strain_history + [$strain_uE] END,
                        v.sensor_acc_x_history = CASE
                            WHEN v.sensor_acc_x_history IS NULL      THEN [$acc_x]
                            WHEN size(v.sensor_acc_x_history) >= $max_history
                                THEN v.sensor_acc_x_history[1..] + [$acc_x]
                            ELSE v.sensor_acc_x_history + [$acc_x] END,
                        v.sensor_acc_y_history = CASE
                            WHEN v.sensor_acc_y_history IS NULL      THEN [$acc_y]
                            WHEN size(v.sensor_acc_y_history) >= $max_history
                                THEN v.sensor_acc_y_history[1..] + [$acc_y]
                            ELSE v.sensor_acc_y_history + [$acc_y] END,
                        v.sensor_acc_z_history = CASE
                            WHEN v.sensor_acc_z_history IS NULL      THEN [$acc_z]
                            WHEN size(v.sensor_acc_z_history) >= $max_history
                                THEN v.sensor_acc_z_history[1..] + [$acc_z]
                            ELSE v.sensor_acc_z_history + [$acc_z] END,
                        v.sensor_timestamp_history = CASE
                            WHEN v.sensor_timestamp_history IS NULL  THEN [$timestamp]
                            WHEN size(v.sensor_timestamp_history) >= $max_history
                                THEN v.sensor_timestamp_history[1..] + [$timestamp]
                            ELSE v.sensor_timestamp_history + [$timestamp] END
                    RETURN v.grid_i AS i
                """
            elif rich and not is_mpu:
                # SG voxel with raw HX711: strain + hx711 only — NO acc/gyro
                base_params["hx711_raw"] = int(rich.get("hx711_raw", 0))
                cypher = """
                    MATCH (v:Voxel {grid_i: $grid_i, grid_j: $grid_j, grid_k: $grid_k})
                    SET v.sensor_id         = $sensor_id,
                        v.sensor_type       = $sensor_type,
                        v.sensor_strain_uE  = $strain_uE,
                        v.sensor_hx711_raw  = $hx711_raw,
                        v.last_updated      = $timestamp,
                        v.sensor_strain_history = CASE
                            WHEN v.sensor_strain_history IS NULL     THEN [$strain_uE]
                            WHEN size(v.sensor_strain_history) >= $max_history
                                THEN v.sensor_strain_history[1..] + [$strain_uE]
                            ELSE v.sensor_strain_history + [$strain_uE] END,
                        v.sensor_timestamp_history = CASE
                            WHEN v.sensor_timestamp_history IS NULL  THEN [$timestamp]
                            WHEN size(v.sensor_timestamp_history) >= $max_history
                                THEN v.sensor_timestamp_history[1..] + [$timestamp]
                            ELSE v.sensor_timestamp_history + [$timestamp] END
                    RETURN v.grid_i AS i
                """
            else:
                # No rich data: strain-only fallback
                cypher = """
                    MATCH (v:Voxel {grid_i: $grid_i, grid_j: $grid_j, grid_k: $grid_k})
                    SET v.sensor_id         = $sensor_id,
                        v.sensor_type       = $sensor_type,
                        v.sensor_strain_uE  = $strain_uE,
                        v.last_updated      = $timestamp,
                        v.sensor_strain_history = CASE
                            WHEN v.sensor_strain_history IS NULL     THEN [$strain_uE]
                            WHEN size(v.sensor_strain_history) >= $max_history
                                THEN v.sensor_strain_history[1..] + [$strain_uE]
                            ELSE v.sensor_strain_history + [$strain_uE] END,
                        v.sensor_timestamp_history = CASE
                            WHEN v.sensor_timestamp_history IS NULL  THEN [$timestamp]
                            WHEN size(v.sensor_timestamp_history) >= $max_history
                                THEN v.sensor_timestamp_history[1..] + [$timestamp]
                            ELSE v.sensor_timestamp_history + [$timestamp] END
                    RETURN v.grid_i AS i
                """

            # For SG sensors, write the same reading to every voxel in the strut
            # (the gauge measures the average over its bonded length)
            strut = pos.get("strut_voxels")  # list of [i,j,k] or None
            voxels_to_update = strut if strut else [[pos["grid_i"], pos["grid_j"], pos["grid_k"]]]

            for vox in voxels_to_update:
                params = dict(base_params)
                params["grid_i"] = int(vox[0])
                params["grid_j"] = int(vox[1])
                params["grid_k"] = int(vox[2])
                result = session.run(cypher, params)
                if result.single():
                    updated_count += 1

        # Overwrite (not append) the SensorStream node with the latest values
        stream_params: Dict[str, Any] = {
            "timestamp": timestamp,
            "s1": float(sensor_readings[0]),
            "s2": float(sensor_readings[1]),
            "s3": float(sensor_readings[2]),
            "s4": float(sensor_readings[3]),
        }
        # Store full rich data on the stream node too if available
        r1 = (rich_sensors or {}).get("S1") or {}
        r2 = (rich_sensors or {}).get("S2") or {}
        stream_params.update({
            "s1_hx711":  int(r1.get("hx711_raw", 0)),
            "s1_acc_x":  float(r1.get("acc_x", 0)),
            "s1_acc_y":  float(r1.get("acc_y", 0)),
            "s1_acc_z":  float(r1.get("acc_z", 0)),
            "s1_gyro_x": float(r1.get("gyro_x", 0)),
            "s1_gyro_y": float(r1.get("gyro_y", 0)),
            "s1_gyro_z": float(r1.get("gyro_z", 0)),
            "s2_hx711":  int(r2.get("hx711_raw", 0)),
            "s2_acc_x":  float(r2.get("acc_x", 0)),
            "s2_acc_y":  float(r2.get("acc_y", 0)),
            "s2_acc_z":  float(r2.get("acc_z", 0)),
            "s2_gyro_x": float(r2.get("gyro_x", 0)),
            "s2_gyro_y": float(r2.get("gyro_y", 0)),
            "s2_gyro_z": float(r2.get("gyro_z", 0)),
        })
        session.run("""
            MERGE (s:SensorStream {id: 'global'})
            SET s.last_timestamp = $timestamp,
                s.last_updated   = $timestamp,
                s.S1             = $s1,
                s.S2             = $s2,
                s.S3             = $s3,
                s.S4             = $s4,
                s.S1_hx711_raw   = $s1_hx711,
                s.S1_acc_x       = $s1_acc_x,
                s.S1_acc_y       = $s1_acc_y,
                s.S1_acc_z       = $s1_acc_z,
                s.S1_gyro_x      = $s1_gyro_x,
                s.S1_gyro_y      = $s1_gyro_y,
                s.S1_gyro_z      = $s1_gyro_z,
                s.S2_hx711_raw   = $s2_hx711,
                s.S2_acc_x       = $s2_acc_x,
                s.S2_acc_y       = $s2_acc_y,
                s.S2_acc_z       = $s2_acc_z,
                s.S2_gyro_x      = $s2_gyro_x,
                s.S2_gyro_y      = $s2_gyro_y,
                s.S2_gyro_z      = $s2_gyro_z
        """, stream_params)

    driver.close()

    return {
        "success": True,
        "sensors_updated": updated_count,
        "timestamp": timestamp,
        "readings": dict(zip(sensor_ids, sensor_readings)),
    }


def get_sensor_readings(sensor_positions: Dict[str, Dict]) -> List[Optional[float]]:
    """
    Retrieve current sensor readings (strain_uE) from Neo4j.
    For SG sensors with a strut, returns the average strain across all strut voxels.
    """
    conn_info = Settings.get_connection_info()
    driver = GraphDatabase.driver(
        conn_info["uri"],
        auth=(conn_info["user"], conn_info["password"])
    )

    sensor_ids = ["S1", "S2", "S3", "S4"]
    readings: List[Optional[float]] = []

    with driver.session(database=conn_info.get("database")) as session:
        for sensor_id in sensor_ids:
            pos = sensor_positions.get(sensor_id)
            if pos is None:
                readings.append(None)
                continue

            strut = pos.get("strut_voxels")
            if strut:
                # Average strain_uE across all strut voxels (SG measurement area)
                vals = []
                for vox in strut:
                    res = session.run("""
                        MATCH (v:Voxel {grid_i: $grid_i, grid_j: $grid_j, grid_k: $grid_k})
                        RETURN v.sensor_strain_uE AS strain_uE
                    """, {"grid_i": int(vox[0]), "grid_j": int(vox[1]), "grid_k": int(vox[2])})
                    rec = res.single()
                    if rec and rec["strain_uE"] is not None:
                        vals.append(float(rec["strain_uE"]))
                readings.append(sum(vals) / len(vals) if vals else None)
            else:
                result = session.run("""
                    MATCH (v:Voxel {grid_i: $grid_i, grid_j: $grid_j, grid_k: $grid_k})
                    RETURN v.sensor_strain_uE AS strain_uE
                """, {"grid_i": pos["grid_i"], "grid_j": pos["grid_j"], "grid_k": pos["grid_k"]})
                record = result.single()
                readings.append(
                    float(record["strain_uE"]) if record and record["strain_uE"] is not None else None
                )

    driver.close()
    return readings


def reset_sensor_stream() -> None:
    """Delete the SensorStream node (clean slate on pipeline restart)."""
    conn_info = Settings.get_connection_info()
    driver = GraphDatabase.driver(
        conn_info["uri"],
        auth=(conn_info["user"], conn_info["password"])
    )
    with driver.session(database=conn_info.get("database")) as session:
        session.run("MATCH (s:SensorStream) DETACH DELETE s")
    driver.close()
