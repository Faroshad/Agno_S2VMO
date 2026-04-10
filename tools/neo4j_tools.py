#!/usr/bin/env python3
"""
Neo4j Tools for Agno GraphRAG using Intelligent Query Generation

Uses LLM to dynamically generate Cypher queries based on natural language questions.
This provides unlimited flexibility compared to predefined query functions.

Features:
- Dynamic Cypher generation for any question
- Handles complex multi-condition queries
- Adapts to new query patterns automatically
- Only 2 tools instead of 8+ predefined functions
"""

from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
from agno.agent import Toolkit
from openai import OpenAI
# Support both package-relative and absolute imports
try:
    from ..config.settings import Settings
except ImportError:
    from config.settings import Settings
import json


# Neo4j connection singleton
_neo4j_driver = None

def _get_driver():
    """Get or create Neo4j driver"""
    global _neo4j_driver
    if _neo4j_driver is None:
        conn_info = Settings.get_connection_info()
        _neo4j_driver = GraphDatabase.driver(
            conn_info["uri"],
            auth=(conn_info["user"], conn_info["password"])
        )
    return _neo4j_driver


def neo4j_ping() -> bool:
    """
    Return True only if Neo4j accepts a session and runs a trivial query.
    Used by /api/neo4j/status — unlike _execute_query, this does NOT swallow errors.
    """
    try:
        conn_info = Settings.get_connection_info()
        driver = _get_driver()
        db = conn_info.get("database")
        if db:
            with driver.session(database=db) as session:
                session.run("RETURN 1 AS ping").consume()
        else:
            with driver.session() as session:
                session.run("RETURN 1 AS ping").consume()
        return True
    except Exception:
        return False


def _execute_query(cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
    """Execute a Cypher query and convert Neo4j objects to dictionaries"""
    conn_info = Settings.get_connection_info()
    db = conn_info.get("database")
    ctx = _get_driver().session(database=db) if db else _get_driver().session()
    with ctx as session:
        try:
            result = session.run(cypher_query, parameters or {})
            records = []
            for record in result:
                record_dict = {}
                for key in record.keys():
                    value = record[key]
                    # Convert Neo4j Node objects to dictionaries
                    if hasattr(value, '__class__') and value.__class__.__name__ == 'Node':
                        record_dict[key] = dict(value.items())
                    else:
                        record_dict[key] = value
                records.append(record_dict)
            return records
        except Exception as e:
            print(f"Query error: {e}")
            return []
    

# Database schema for LLM
DATABASE_SCHEMA = """
DATABASE SCHEMA:

Nodes:
- Voxel
  Properties: 
    - id (int): Unique voxel identifier
    - grid_i, grid_j, grid_k (int): Voxel grid indices
    - x, y, z (float): 3D position coordinates
    - type (string: 'joint' or 'beam'): Structural element type
    - connection_count (int): Number of adjacent voxels
    - ground_connected (boolean): Whether connected to ground
    - ground_level (float or null): Ground level if connected
    
  Sensor Properties (may be null when no sensor has been assigned or MQTT is offline):
    PHYSICAL SENSOR TAGS (set on every write by the MQTT pipeline):
    - sensor_id   (string): 'S1', 'S2', 'S3', or 'S4'
    - sensor_type (string): 'MPU' (accelerometer+gyro) or 'SG' (strain gauge only)

    ⚠️ HOW SENSORS ARE MODELED (critical for Cypher):
    - There is NO (:Sensor) node. There is NO relationship like (Voxel)-[:HAS_SENSOR|CONNECTED_TO]->(...)
    - Sensors are ONLY the property **sensor_id** (and related sensor_* fields) on **Voxel** nodes.
    - "Voxels connected to sensors" / "sensor voxels" means: MATCH (v:Voxel) WHERE v.sensor_id IS NOT NULL
    - SG struts tag **multiple** voxels with the SAME sensor_id (e.g. many rows for 'S3').
      • "How many voxels have sensors?" → count(v) with sensor_id IS NOT NULL (can be >4).
      • "How many physical sensors are placed?" → count(DISTINCT v.sensor_id) (expect up to 4).

    LATEST SCALARS (fastest access):
    - sensor_strain_uE (float): Latest strain in microstrains (μE)
    - sensor_hx711_raw (int): Latest raw HX711 load cell reading
    - sensor_acc_x/y/z (float): Acceleration in g — MPU voxels only (null on SG)
    - sensor_gyro_x/y/z (float): Gyroscope in deg/s — MPU voxels only (null on SG)
    - last_updated (string): ISO timestamp of the most recent MQTT write

    TIME-SERIES HISTORY (arrays — one entry per MQTT cycle, capped at 200):
    - sensor_strain_history (array of floats): All strain_uE readings, oldest→newest
    - sensor_acc_x_history, sensor_acc_y_history, sensor_acc_z_history (array of floats)
    - sensor_timestamp_history (array of strings): Matching ISO timestamps

    ⚠️ Use `sensor_strain_uE` (NOT `strain_uE`) in all Cypher queries.
    Use `sensor_strain_history` for trends/spikes; `sensor_strain_uE` for latest value.

    UI / CHATBOT LABELS vs GRAPH (critical when users say "SG1" or "SG2"):
    - Dashboard chart **SG1** (strain gauge 1) = Neo4j `sensor_id` **'S3'** (device 1 gauge → S3 voxel)
    - Dashboard chart **SG2** (strain gauge 2) = Neo4j `sensor_id` **'S4'** (device 2 gauge → S4 voxel)
    - **S1** / **S2** voxels are MPU placements; FEM uses the same strain numbers as S3/S4 but
      questions about "the two strain gauges" always mean **S3 vs S4** histories.

    COMPARING TWO STRAIN TIME SERIES (never use exact `=` on floats):
    - Raw `WHERE sg1 = sg2` will almost always return **no rows** — floating-point noise.
    - "Same value" → use a tolerance in μE: `abs(sg1 - sg2) < 2.0` (tune 1–3 μE).
    - Chart **intersection** (lines cross) → at some index `i` either:
      (a) |sg1[i]-sg2[i]| < tolerance, OR
      (b) the **difference changes sign** between i and i+1: (sg1[i]-sg2[i]) * (sg1[i+1]-sg2[i+1]) < 0
      (lines crossed between samples; equality at a point is a special case).
    - Histories are **aligned by index** when both arrays are updated each MQTT cycle (same length).

    SPIKE DETECTION PATTERN (detect sudden change in last N readings):
    MATCH (v:Voxel) WHERE v.sensor_strain_history IS NOT NULL
      AND size(v.sensor_strain_history) >= 3
    WITH v,
         v.sensor_strain_history[-1] AS latest,
         reduce(acc=0.0, x IN v.sensor_strain_history[-5..] | acc + x) /
             size(v.sensor_strain_history[-5..]) AS recent_avg
    WHERE abs(latest - recent_avg) > 50
    RETURN v.id, v.grid_i, v.grid_j, v.grid_k, latest, recent_avg,
           (latest - recent_avg) AS delta,
           v.sensor_timestamp_history[-1] AS spike_time
    NOTE: Cypher supports negative array indexing: list[-1] = last element,
          list[-5..] = last 5 elements.

  FEM Analysis Properties (SCALAR floats — overwritten each FEM cycle, latest value only):
    - stress_magnitude (float): Von Mises stress magnitude in Pascals (Pa)
    - eps_xx, eps_yy, eps_zz (float): Strain tensor components
    - sigma_xx, sigma_yy, sigma_zz (float): Normal stress components in Pa
    - sigma_xy, sigma_yz, sigma_xz (float): Shear stress components in Pa
    - fem_timestamp (string): ISO timestamp of the last FEM run that updated this voxel
    - last_updated (string): Most recent update timestamp (sensor or FEM)

    🔍 ACCESS RULES (scalars — no array indexing needed):
    - Check voxel has FEM data: WHERE v.stress_magnitude IS NOT NULL
    - Query: MATCH (v:Voxel) WHERE v.stress_magnitude IS NOT NULL
             RETURN v.id, v.stress_magnitude, v.grid_i, v.grid_j, v.grid_k

- FEMAnalysis
  There is EXACTLY ONE row in the graph: MERGE (:FEMAnalysis {analysis_id: 'latest'}).
  It is updated in-place every FEM run — voxel stress fields overwrite; this node accumulates run metadata.

  Properties:
    - analysis_id (string): Always 'latest' for the current session summary node
    - fem_cycle_count (int): Total completed FEM runs since last MQTT connect (increments each pipeline)
      ⚠️ To answer "how many FEM cycles / simulations": RETURN a.fem_cycle_count from this node.
      NEVER use count(a:FEMAnalysis) or count(DISTINCT FEMAnalysis) for cycle count — that is always 1.
    - fem_cycle_timestamps (list of strings): ISO time of each completed FEM run (oldest→newest, capped ~2000)
      Use this for "what periods / when was stress computed" — NOT distinct v.fem_timestamp on voxels
      (all voxels share the same last fem_timestamp after each overwrite).
    - timestamp (string): ISO timestamp of the most recent FEM run (same as last entry in fem_cycle_timestamps)
    - sensor_reading (list of floats): Last sensor microstrain values used for FEM
    - total_voxels (int): Number of solid voxels analyzed
    - status (string): Analysis status ('completed', etc.)
    - avg_stress_magnitude, max_stress_magnitude, min_stress_magnitude (float): Rolled up from voxels

- FEMResult
  Properties:
    - voxel_grid_pos (string): Grid position identifier
    - analysis_id (string): Link to FEMAnalysis
    - timestamp (string): Analysis timestamp
    - eps_xx, eps_yy, eps_zz (floats): Strain components
    - sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz (floats): Stress components
    - stress_magnitude (float): Von Mises stress in Pa

Relationships:
- (Voxel)-[:ADJACENT_TO]->(Voxel) - Spatial neighbors
- (Voxel)-[:HAS_FEM_RESULT]->(FEMResult) - Links voxel to its FEM results
- (FEMAnalysis)-[:PRODUCED_RESULT]->(FEMResult) - Groups results by analysis

Indexes:
- Vector index on Chunk embeddings for semantic search

VOXEL TYPE DEFINITIONS:
- 'joint': Structural connection points (high connectivity, many neighbors)
- 'beam': Linear elements between joints (lower connectivity, 2 neighbors)

GEOMETRIC FEATURE DEFINITIONS (derived from connection_count):
The `connection_count` on each Voxel is the number of solid voxels immediately
adjacent (sharing a face) in the 3D grid.  Use it to identify structural zones:

- Surface voxel  : connection_count < 6
  → Has at least one exposed face (touching empty space or the boundary)
  → Query: MATCH (v:Voxel) WHERE v.connection_count < 6 RETURN count(v)

- Interior voxel : connection_count = 6
  → Fully surrounded by other solid voxels (buried inside the structure)
  → Query: MATCH (v:Voxel) WHERE v.connection_count = 6 RETURN count(v)

- Edge/corner    : connection_count <= 3
  → Voxels at sharp edges or corners, very exposed
  → Query: MATCH (v:Voxel) WHERE v.connection_count <= 3 RETURN count(v)

- Highly connected (structural joints): connection_count > 10
  → Voxels where many struts meet — load-bearing nodes
  → Query: MATCH (v:Voxel) WHERE v.connection_count > 10 RETURN count(v)

For a dome (thin shell), the vast majority of voxels ARE surface voxels
(connection_count < 6) because the shell is only 1-2 voxels thick.

NOTE: `ground_connected` (boolean) marks voxels that are structurally
connected to the base/ground.  `ground_level` gives the height.
Use `ground_connected = true` to identify base/foundation voxels.

EXAMPLE QUERIES:

1. Count all voxels:
   MATCH (v:Voxel) RETURN count(v) as count

2. Find voxels by type:
   MATCH (v:Voxel {type: 'joint'}) 
   RETURN v.id, v.x, v.y, v.z, v.connection_count
   LIMIT 10

3. Find neighbors of a voxel:
   MATCH (v:Voxel {id: 5})-[:ADJACENT_TO]-(neighbor:Voxel)
   RETURN neighbor.id, neighbor.x, neighbor.y, neighbor.z, neighbor.type

4. Find ground-connected voxels:
   MATCH (v:Voxel {ground_connected: true})
   RETURN v.id, v.x, v.y, v.z, v.type
   LIMIT 10

5. Find most connected voxel:
   MATCH (v:Voxel)
   RETURN v.id, v.connection_count, v.type
   ORDER BY v.connection_count DESC
   LIMIT 1

6. Find voxels with coordinate constraints:
   MATCH (v:Voxel)
   WHERE v.x > -2.0 AND v.x < 2.0
   RETURN v.id, v.x, v.y, v.z, v.type
   LIMIT 10

7. Find voxels with sensor data (latest reading):
   MATCH (v:Voxel)
   WHERE v.sensor_strain_uE IS NOT NULL
   RETURN v.id, v.sensor_strain_uE, v.x, v.y, v.z, v.last_updated
   LIMIT 10

8. Find voxels with high strain (latest reading):
   MATCH (v:Voxel)
   WHERE v.sensor_strain_uE > 100
   RETURN v.id, v.sensor_strain_uE, v.x, v.y, v.z, v.type
   ORDER BY v.sensor_strain_uE DESC
   LIMIT 10

9. Find voxels with FEM stress data:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL
   RETURN v.id, v.stress_magnitude, v.x, v.y, v.z, v.type
   ORDER BY v.stress_magnitude DESC
   LIMIT 10

GEOMETRIC QUERIES:

28. Count surface voxels (exposed to empty space — has unexposed face):
   MATCH (v:Voxel) WHERE v.connection_count < 6
   RETURN count(v) as surface_voxels

29. Count interior voxels (fully surrounded):
   MATCH (v:Voxel) WHERE v.connection_count = 6
   RETURN count(v) as interior_voxels

30. Surface voxel distribution by exposure level:
   MATCH (v:Voxel)
   RETURN
     CASE
       WHEN v.connection_count <= 2 THEN 'Tip/Corner (<=2 neighbors)'
       WHEN v.connection_count <= 3 THEN 'Edge (3 neighbors)'
       WHEN v.connection_count <= 4 THEN 'Surface (4 neighbors)'
       WHEN v.connection_count = 5  THEN 'Near-surface (5 neighbors)'
       WHEN v.connection_count = 6  THEN 'Interior (6 neighbors)'
       ELSE 'Highly connected (>6)'
     END as zone,
     count(*) as count
   ORDER BY count DESC

31. Find ground-level / base voxels:
   MATCH (v:Voxel) WHERE v.ground_connected = true
   RETURN count(v) as base_voxels

32. Find voxels by height zone (z coordinate):
   MATCH (v:Voxel)
   RETURN round(v.z * 2) / 2 as height_zone, count(v) as voxels_at_height
   ORDER BY height_zone

SENSOR HISTORY / TIME-SERIES QUERIES:

33. Get full strain history for sensor voxels:
   MATCH (v:Voxel)
   WHERE v.sensor_strain_history IS NOT NULL
   RETURN v.id, v.grid_i, v.grid_j, v.grid_k,
          v.sensor_strain_history as strain_readings,
          v.sensor_timestamp_history as timestamps,
          size(v.sensor_strain_history) as reading_count

34. Detect strain spikes (latest reading deviates >50μE from recent average):
   MATCH (v:Voxel)
   WHERE v.sensor_strain_history IS NOT NULL
     AND size(v.sensor_strain_history) >= 3
   WITH v,
        v.sensor_strain_history[-1] AS latest,
        reduce(acc=0.0, x IN v.sensor_strain_history[-5..] | acc + x) /
            size(v.sensor_strain_history[-5..]) AS recent_avg
   WHERE abs(latest - recent_avg) > 50
   RETURN v.id, v.grid_i, v.grid_j, v.grid_k,
          latest, recent_avg, (latest - recent_avg) AS delta,
          v.sensor_timestamp_history[-1] AS spike_time
   ORDER BY abs(latest - recent_avg) DESC

35. Get min, max, average strain from history for each sensor voxel:
   MATCH (v:Voxel)
   WHERE v.sensor_strain_history IS NOT NULL
   WITH v,
        reduce(mn=v.sensor_strain_history[0], x IN v.sensor_strain_history |
            CASE WHEN x < mn THEN x ELSE mn END) AS min_strain,
        reduce(mx=v.sensor_strain_history[0], x IN v.sensor_strain_history |
            CASE WHEN x > mx THEN x ELSE mx END) AS max_strain,
        reduce(s=0.0, x IN v.sensor_strain_history | s + x) /
            size(v.sensor_strain_history) AS avg_strain
   RETURN v.id, v.grid_i, v.grid_j, v.grid_k,
          min_strain, max_strain, avg_strain,
          (max_strain - min_strain) AS strain_range,
          size(v.sensor_strain_history) AS reading_count

36. Detect acceleration anomalies (|acc_z| deviates from 1g baseline):
   MATCH (v:Voxel)
   WHERE v.sensor_acc_z_history IS NOT NULL
     AND size(v.sensor_acc_z_history) >= 3
   WITH v,
        v.sensor_acc_z_history[-1] AS latest_az,
        reduce(acc=0.0, x IN v.sensor_acc_z_history[-5..] | acc + x) /
            size(v.sensor_acc_z_history[-5..]) AS avg_az
   WHERE abs(latest_az - 1.0) > 0.1
   RETURN v.id, v.grid_i, v.grid_j, v.grid_k,
          latest_az, avg_az,
          v.sensor_timestamp_history[-1] AS event_time

37. SG1 vs SG2 (chart) = S3 vs S4 (Neo4j): find times when strains were ~equal (μE tolerance).
   Use one voxel per sensor (struts may tag multiple voxels — histories are identical):
   MATCH (v3:Voxel {sensor_id: 'S3'})
   WHERE v3.sensor_strain_history IS NOT NULL
   WITH v3 LIMIT 1
   MATCH (v4:Voxel {sensor_id: 'S4'})
   WHERE v4.sensor_strain_history IS NOT NULL
     AND size(v3.sensor_strain_history) = size(v4.sensor_strain_history)
   WITH v3, v4 LIMIT 1
   WITH v3, v4, range(0, size(v3.sensor_strain_history) - 1) AS idx
   UNWIND idx AS i
   WITH v3.sensor_timestamp_history[i] AS ts,
        v3.sensor_strain_history[i] AS sg1_uE,
        v4.sensor_strain_history[i] AS sg2_uE
   WHERE abs(sg1_uE - sg2_uE) < 2.0
   RETURN ts, sg1_uE, sg2_uE, abs(sg1_uE - sg2_uE) AS abs_diff_uE
   LIMIT 30

38. Detect line crossings between SG1 and SG2 (difference changes sign between consecutive samples):
   MATCH (v3:Voxel {sensor_id: 'S3'})
   WHERE v3.sensor_strain_history IS NOT NULL AND size(v3.sensor_strain_history) >= 2
   WITH v3 LIMIT 1
   MATCH (v4:Voxel {sensor_id: 'S4'})
   WHERE v4.sensor_strain_history IS NOT NULL
     AND size(v3.sensor_strain_history) = size(v4.sensor_strain_history)
   WITH v3, v4 LIMIT 1
   WITH v3, v4, range(0, size(v3.sensor_strain_history) - 2) AS idx
   UNWIND idx AS i
   WITH v3.sensor_timestamp_history[i] AS ts_start,
        v3.sensor_strain_history[i] - v4.sensor_strain_history[i] AS d0,
        v3.sensor_strain_history[i + 1] - v4.sensor_strain_history[i + 1] AS d1
   WHERE d0 * d1 < 0.0
   RETURN ts_start AS crossing_after_time,
          d0 AS diff_before_uE, d1 AS diff_after_uE
   LIMIT 20

10. Find closest voxel to a point or another voxel:
   MATCH (target:Voxel {id: 5})
   MATCH (v:Voxel) WHERE v.id <> 5
   RETURN v.id, v.type, 
          point.distance(point({x: target.x, y: target.y, z: target.z}), 
                         point({x: v.x, y: v.y, z: v.z})) as distance
   ORDER BY distance
   LIMIT 1
   
   NOTE: Use point.distance() NOT distance() (deprecated in newer Neo4j versions)

FEM STRESS/STRAIN QUERIES (SCALAR values — no array indexing):

11. Get maximum stress value:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL
   RETURN v.grid_i, v.grid_j, v.grid_k, v.x, v.y, v.z, v.stress_magnitude
   ORDER BY v.stress_magnitude DESC
   LIMIT 1

12. Get average stress statistics:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL
   RETURN avg(v.stress_magnitude) as avg_Pa,
          min(v.stress_magnitude) as min_Pa,
          max(v.stress_magnitude) as max_Pa,
          count(v) as voxel_count

13. Find high stress regions (>500 Pa):
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND v.stress_magnitude > 500
   RETURN v.grid_i, v.grid_j, v.grid_k, v.x, v.y, v.z, v.stress_magnitude as stress_Pa
   ORDER BY v.stress_magnitude DESC
   LIMIT 20

14. Get strain components for a voxel:
   MATCH (v:Voxel {grid_i: 10, grid_j: 20, grid_k: 30})
   WHERE v.eps_xx IS NOT NULL
   RETURN v.grid_i, v.grid_j, v.grid_k, v.eps_xx, v.eps_yy, v.eps_zz

15. Get stress tensor components:
   MATCH (v:Voxel)
   WHERE v.sigma_xx IS NOT NULL
   RETURN v.grid_i, v.grid_j, v.grid_k,
          v.sigma_xx, v.sigma_yy, v.sigma_zz
   LIMIT 10

16. Get last FEM timestamp for a voxel:
   MATCH (v:Voxel {grid_i: 10, grid_j: 20, grid_k: 30})
   RETURN v.grid_i, v.grid_j, v.grid_k, v.stress_magnitude, v.fem_timestamp

17. Get the single FEMAnalysis node (latest cycle summary + run counts):
   MATCH (a:FEMAnalysis {analysis_id: 'latest'})
   RETURN a.analysis_id, a.fem_cycle_count, a.timestamp, a.total_voxels, a.status,
          a.avg_stress_magnitude, a.max_stress_magnitude, a.min_stress_magnitude,
          size(coalesce(a.fem_cycle_timestamps, [])) AS recorded_cycle_times

39. How many FEM simulation cycles have completed? (NOT count(FEMAnalysis) — that is always 1)
   MATCH (a:FEMAnalysis {analysis_id: 'latest'})
   RETURN coalesce(a.fem_cycle_count, 0) AS fem_cycles_completed

40. When was stress / FEM computed? (historical timestamps — use FEMAnalysis list, not voxels)
   MATCH (a:FEMAnalysis {analysis_id: 'latest'})
   RETURN a.fem_cycle_timestamps AS stress_computed_at_timestamps

41. How many voxel rows are tagged with a sensor?
   MATCH (v:Voxel) WHERE v.sensor_id IS NOT NULL
   RETURN count(v) AS voxels_with_sensor_property

42. How many distinct sensor IDs appear on voxels? (physical channels, usually ≤4)
   MATCH (v:Voxel) WHERE v.sensor_id IS NOT NULL
   RETURN count(DISTINCT v.sensor_id) AS distinct_sensor_ids

18. Count voxels with FEM data:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL
   RETURN count(v) as voxels_with_fem_data

19. Stress distribution by category:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL
   RETURN 
     CASE 
       WHEN v.stress_magnitude < 1000 THEN 'Low (<1kPa)'
       WHEN v.stress_magnitude < 2000 THEN 'Medium (1-2kPa)'
       WHEN v.stress_magnitude < 3000 THEN 'High (2-3kPa)'
       ELSE 'Critical (>3kPa)'
     END as category,
     count(*) as voxel_count
   ORDER BY voxel_count DESC

SAFETY ASSESSMENT QUERIES (Multi-step reasoning):

20. Comprehensive safety statistics:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL
   WITH count(v) as total_voxels,
        avg(v.stress_magnitude) as avg_stress,
        min(v.stress_magnitude) as min_stress,
        max(v.stress_magnitude) as max_stress,
        sum(CASE WHEN v.stress_magnitude > 3000 THEN 1 ELSE 0 END) as critical_count,
        sum(CASE WHEN v.stress_magnitude > 2000 AND v.stress_magnitude <= 3000 THEN 1 ELSE 0 END) as high_count,
        sum(CASE WHEN v.stress_magnitude > 1000 AND v.stress_magnitude <= 2000 THEN 1 ELSE 0 END) as medium_count
   RETURN total_voxels, avg_stress, min_stress, max_stress,
          critical_count, high_count, medium_count,
          toFloat(critical_count) / total_voxels * 100 as critical_percentage

21. Identify critical danger zones:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND v.stress_magnitude > 3000
   RETURN v.grid_i, v.grid_j, v.grid_k, v.x, v.y, v.z,
          v.stress_magnitude as stress_Pa, v.type, v.last_updated as timestamp
   ORDER BY v.stress_magnitude DESC
   LIMIT 20

22. Find voxels under tension (negative eps_zz):
   MATCH (v:Voxel)
   WHERE v.eps_zz IS NOT NULL AND v.eps_zz < 0
   RETURN v.grid_i, v.grid_j, v.grid_k, v.x, v.y, v.z,
          v.eps_zz, v.stress_magnitude,
          CASE 
            WHEN percent_change > 20 THEN 'CRITICAL INCREASE'
            WHEN percent_change > 0 THEN 'Increasing'
            WHEN percent_change < -20 THEN 'Major Decrease'
            ELSE 'Decreasing'
          END as trend
   ORDER BY abs(percent_change) DESC
   LIMIT 10

23. Find stress hotspot clusters (nearby high-stress voxels):
   MATCH (v1:Voxel)-[:ADJACENT_TO]-(v2:Voxel)
   WHERE v1.stress_magnitude IS NOT NULL AND v1.stress_magnitude > 2500
     AND v2.stress_magnitude IS NOT NULL AND v2.stress_magnitude > 2500
   RETURN v1.grid_i, v1.grid_j, v1.grid_k, v1.stress_magnitude as stress1,
          v2.grid_i, v2.grid_j, v2.grid_k, v2.stress_magnitude as stress2
   LIMIT 10

24. Get most critical voxel:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL
   RETURN v.grid_i, v.grid_j, v.grid_k, v.x, v.y, v.z,
          v.stress_magnitude as stress_Pa, v.fem_timestamp
   ORDER BY v.stress_magnitude DESC
   LIMIT 1

25. Monitoring coverage analysis:
   MATCH (v:Voxel)
   WITH count(v) as total_voxels
   MATCH (v2:Voxel)
   WHERE v2.stress_magnitude IS NOT NULL
   WITH total_voxels, count(v2) as monitored_voxels
   RETURN total_voxels, monitored_voxels,
          total_voxels - monitored_voxels as unmonitored_voxels,
          toFloat(monitored_voxels) / total_voxels * 100 as coverage_percentage

26. FEM analysis summary (latest cycle):
   MATCH (a:FEMAnalysis {analysis_id: 'latest'})
   RETURN a.analysis_id, a.timestamp, a.total_voxels, a.status,
          a.avg_stress_magnitude, a.max_stress_magnitude

27. Regional stress analysis (identify affected regions):
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND v.stress_magnitude > 2000
   RETURN round(v.x) as region_x, round(v.y) as region_y, round(v.z) as region_z,
          avg(v.stress_magnitude) as avg_regional_stress,
          max(v.stress_magnitude) as max_regional_stress,
          count(v) as voxel_count
   ORDER BY max_regional_stress DESC

IMPORTANT RULES:
- Always specify which properties to return (e.g., v.id, v.x)
- NEVER use "RETURN v" or "RETURN n" (returns Node objects)
- Use LIMIT to avoid returning too many results
- For coordinates, use WHERE clauses with comparison operators
- Property names are case-sensitive
- For comparisons (>, <, >=, <=), use WHERE clause, NOT in curly braces
  WRONG: MATCH (v:Voxel {connection_count > 5})
  RIGHT: MATCH (v:Voxel) WHERE v.connection_count > 5
- Exact matches can use curly braces: MATCH (v:Voxel {type: 'joint'})
- For distance calculations, ALWAYS use point.distance(), NEVER distance()
  WRONG: distance(point(...), point(...))
  RIGHT: point.distance(point(...), point(...))
- Sensor data uses property `sensor_strain_uE` (NOT `strain_uE`)
- FEM data (stress_magnitude, eps_xx, sigma_xx) are SCALAR floats — never arrays

🚨 CRITICAL FEM QUERY RULES:
- FEM properties are SCALAR floats written each cycle (overwrite, not append)
- CORRECT: WHERE v.stress_magnitude IS NOT NULL
- WRONG:   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
- WRONG:   v.stress_magnitude[size(v.stress_magnitude)-1]   ← NEVER use array syntax!
- Stress values are in Pascals (Pa)
- Use grid_i, grid_j, grid_k for voxel identification
"""


def _validate_and_fix_query(cypher_query: str, error_message: str, reasoning: dict, original_question: str, client: OpenAI, openai_config: dict) -> str:
    """
    Validate Cypher query and attempt to fix syntax errors.
    
    Args:
        cypher_query: The generated Cypher query
        error_message: Error message from Neo4j
        reasoning: Reasoning analysis from first attempt
        original_question: Original user question
        client: OpenAI client
        openai_config: OpenAI configuration
        
    Returns:
        Fixed Cypher query
    """
    print(f"\n⚠️  Query validation failed, attempting to fix...")
    print(f"   Error: {error_message[:100]}...")
    
    fix_prompt = f"""You are a Cypher query expert. The following query has a syntax error.

ORIGINAL QUESTION: {original_question}

PREVIOUS REASONING:
{json.dumps(reasoning, indent=2)}

GENERATED QUERY (WITH ERROR):
{cypher_query}

ERROR MESSAGE:
{error_message}

COMMON ERRORS TO FIX:
1. Variable scope loss in WITH clause - ALWAYS pass variables: WITH v, calculated_value
2. Multiple MATCH statements - combine into one or use proper WITH chain
3. Missing variable definitions - ensure all variables are defined before use

Please generate a CORRECTED Cypher query that fixes this error.
Return ONLY the corrected Cypher query, no explanations."""

    try:
        response = client.chat.completions.create(
            model=openai_config["model"],
            messages=[
                {"role": "system", "content": "You are a Cypher query expert who fixes syntax errors."},
                {"role": "user", "content": fix_prompt}
            ],
            temperature=0.0
        )
        
        fixed_query = response.choices[0].message.content.strip()
        
        # Clean up the query
        if fixed_query.startswith("```"):
            lines = fixed_query.split("\n")
            fixed_query = "\n".join([l for l in lines if not l.startswith("```")])
            fixed_query = fixed_query.strip()
        
        print(f"🔧 Generated fixed query")
        return fixed_query
        
    except Exception as e:
        print(f"❌ Failed to fix query: {e}")
        return cypher_query  # Return original if fix fails


# ── Safe fallback queries for the most common intents ──────────────────────
# When the LLM-generated Cypher fails, we attempt these exact, tested queries
# before giving up.  Key: lowercase keyword found in the natural-language query.
_SAFE_FALLBACKS: dict = {
    "highest stress":
        "MATCH (v:Voxel) WHERE v.stress_magnitude IS NOT NULL "
        "RETURN v.id, v.stress_magnitude, v.x, v.y, v.z, v.type "
        "ORDER BY v.stress_magnitude DESC LIMIT 10",
    "max stress":
        "MATCH (v:Voxel) WHERE v.stress_magnitude IS NOT NULL "
        "RETURN v.id, v.stress_magnitude, v.x, v.y, v.z, v.type "
        "ORDER BY v.stress_magnitude DESC LIMIT 10",
    "stress voxel":
        "MATCH (v:Voxel) WHERE v.stress_magnitude IS NOT NULL "
        "RETURN v.id, v.stress_magnitude, v.x, v.y, v.z, v.type "
        "ORDER BY v.stress_magnitude DESC LIMIT 10",
    "fem cycle":
        "MATCH (a:FEMAnalysis {analysis_id:'latest'}) "
        "RETURN a.fem_cycle_count AS cycles, "
        "a.max_stress_magnitude AS max_stress, "
        "a.avg_stress_magnitude AS avg_stress, "
        "a.min_stress_magnitude AS min_stress, "
        "a.timestamp AS last_run, a.fem_cycle_timestamps AS timestamps",
    "recent fem":
        "MATCH (a:FEMAnalysis {analysis_id:'latest'}) "
        "RETURN a.fem_cycle_count AS cycles, "
        "a.max_stress_magnitude AS max_stress, "
        "a.avg_stress_magnitude AS avg_stress, "
        "a.timestamp AS last_run, a.fem_cycle_timestamps AS timestamps",
    "fem histor":
        "MATCH (a:FEMAnalysis {analysis_id:'latest'}) "
        "RETURN a.fem_cycle_count AS cycles, a.fem_cycle_timestamps AS timestamps, "
        "a.max_stress_magnitude AS max_stress, a.avg_stress_magnitude AS avg_stress",
    "sensor reading":
        "MATCH (v:Voxel) WHERE v.sensor_id IS NOT NULL "
        "RETURN v.sensor_id, v.sensor_type, v.sensor_strain_uE, "
        "v.sensor_acc_x, v.sensor_acc_y, v.sensor_acc_z, v.last_updated "
        "ORDER BY v.sensor_id",
    "sensor data":
        "MATCH (v:Voxel) WHERE v.sensor_id IS NOT NULL "
        "RETURN v.sensor_id, v.sensor_type, v.sensor_strain_uE, v.last_updated "
        "ORDER BY v.sensor_id",
    "structural safety":
        "MATCH (v:Voxel) WHERE v.stress_magnitude IS NOT NULL "
        "RETURN count(v) AS total_voxels, "
        "avg(v.stress_magnitude) AS avg_stress, "
        "max(v.stress_magnitude) AS max_stress, "
        "min(v.stress_magnitude) AS min_stress, "
        "sum(CASE WHEN v.stress_magnitude > 3000 THEN 1 ELSE 0 END) AS critical_count, "
        "sum(CASE WHEN v.stress_magnitude > 2000 AND v.stress_magnitude <= 3000 THEN 1 ELSE 0 END) AS warning_count",
    "safety":
        "MATCH (v:Voxel) WHERE v.stress_magnitude IS NOT NULL "
        "RETURN count(v) AS total_voxels, "
        "avg(v.stress_magnitude) AS avg_stress, "
        "max(v.stress_magnitude) AS max_stress, "
        "sum(CASE WHEN v.stress_magnitude > 3000 THEN 1 ELSE 0 END) AS critical_count",
    "voxel count":
        "MATCH (v:Voxel) RETURN count(v) AS total_voxels, "
        "sum(CASE WHEN v.type='joint' THEN 1 ELSE 0 END) AS joints, "
        "sum(CASE WHEN v.type='beam'  THEN 1 ELSE 0 END) AS beams, "
        "sum(CASE WHEN v.stress_magnitude IS NOT NULL THEN 1 ELSE 0 END) AS with_fem",
    "how many voxel":
        "MATCH (v:Voxel) RETURN count(v) AS total_voxels",
    "stress statistic":
        "MATCH (v:Voxel) WHERE v.stress_magnitude IS NOT NULL "
        "RETURN count(v) AS voxels_with_stress, "
        "avg(v.stress_magnitude) AS avg_stress, "
        "max(v.stress_magnitude) AS max_stress, "
        "min(v.stress_magnitude) AS min_stress",
}


def _try_safe_fallback(natural_language_query: str) -> Optional[List[Dict]]:
    """
    Try a pre-validated fallback Cypher for the recognised intent.
    Returns results list (may be empty) if a matching fallback was found and ran,
    or None if no matching fallback exists.
    """
    q = natural_language_query.lower()
    for keyword, cypher in _SAFE_FALLBACKS.items():
        if keyword in q:
            print(f"🔄 [fallback] Trying safe query for intent='{keyword}'")
            try:
                results = _execute_query(cypher)
                print(f"✅ [fallback] Got {len(results)} rows")
                return results
            except Exception as exc:
                print(f"⚠️  [fallback] Also failed: {exc}")
                return []   # signal "fallback attempted but empty/error"
    return None  # no fallback for this intent


def _auto_diagnose() -> dict:
    """
    Run a quick set of diagnostic queries and return a compact state dict.
    Called automatically when intelligent_query_neo4j returns empty / fails.
    Never raises — always returns a dict (possibly with 'error' keys).
    """
    diag = {}
    _q = _execute_query   # alias for brevity

    def _safe(key, cypher, extractor):
        try:
            rows = _q(cypher)
            diag[key] = extractor(rows)
        except Exception as exc:
            diag[key] = f"error: {exc}"

    _safe("total_voxels",
          "MATCH (v:Voxel) RETURN count(v) AS n",
          lambda r: r[0]["n"] if r else 0)

    _safe("voxels_with_fem",
          "MATCH (v:Voxel) WHERE v.stress_magnitude IS NOT NULL RETURN count(v) AS n",
          lambda r: r[0]["n"] if r else 0)

    _safe("fem_analysis",
          "MATCH (a:FEMAnalysis {analysis_id:'latest'}) "
          "RETURN a.fem_cycle_count, a.timestamp, "
          "a.max_stress_magnitude, a.avg_stress_magnitude",
          lambda r: r[0] if r else None)

    _safe("assigned_sensors",
          "MATCH (v:Voxel) WHERE v.sensor_id IS NOT NULL "
          "RETURN count(DISTINCT v.sensor_id) AS sensors",
          lambda r: r[0]["sensors"] if r else 0)

    _safe("latest_sensor_readings",
          "MATCH (v:Voxel) WHERE v.sensor_strain_uE IS NOT NULL "
          "RETURN v.sensor_id, v.sensor_strain_uE, v.last_updated "
          "ORDER BY v.sensor_id",
          lambda r: r)

    _safe("voxels_with_sensors",
          "MATCH (v:Voxel) WHERE v.sensor_id IS NOT NULL RETURN count(v) AS n",
          lambda r: r[0]["n"] if r else 0)

    # Derive a plain-English state summary for the agent
    fem_ok = isinstance(diag.get("voxels_with_fem"), int) and diag["voxels_with_fem"] > 0
    sensors_ok = isinstance(diag.get("assigned_sensors"), int) and diag["assigned_sensors"] > 0
    fem_info = diag.get("fem_analysis")
    cycle_count = fem_info.get("a.fem_cycle_count", 0) if isinstance(fem_info, dict) else 0

    summary_parts = []
    if not fem_ok:
        summary_parts.append(
            "FEM stress data is NOT in Neo4j yet "
            f"(0 voxels have stress_magnitude; "
            f"FEM cycles completed: {cycle_count}). "
            "The simulation may not have run, or voxels may not have been written to Neo4j."
        )
    else:
        summary_parts.append(
            f"FEM data is available for {diag['voxels_with_fem']} voxels "
            f"({cycle_count} cycles completed)."
        )

    if not sensors_ok:
        summary_parts.append(
            "No sensor readings in Neo4j yet "
            "(sensors not assigned or MQTT pipeline offline)."
        )
    else:
        summary_parts.append(
            f"{diag['assigned_sensors']} sensor(s) have readings in Neo4j."
        )

    diag["state_summary"] = " ".join(summary_parts)
    return diag


def probe_database_state() -> str:
    """
    Quick diagnostic snapshot of the Neo4j database.

    Call this when:
    - Other queries return empty results and you need to understand why
    - The user asks what data is currently available
    - You need to calibrate your answer based on what is actually in the graph

    Returns:
        JSON with counts, FEM status, sensor status and a plain-English state_summary.
    """
    try:
        return json.dumps(_auto_diagnose(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "state_summary": f"Diagnostic failed: {e}"}, indent=2)


def intelligent_query_neo4j(natural_language_query: str) -> str:
    """
    Intelligently query Neo4j by generating Cypher based on natural language.
    
    Enhanced with multi-step reasoning chain and automatic error recovery:
    1. Analyze question intent and context
    2. Extract specific entities (voxel IDs, conditions)
    3. Break down complex questions into logical steps
    4. Generate precise Cypher query
    5. Validate query execution
    6. Auto-retry with error feedback if syntax error occurs
        
        Args:
        natural_language_query: User's question in natural language
            
        Returns:
        Query results as JSON string with reasoning trace
    """
    # Get OpenAI client
    openai_config = Settings.get_openai_config()
    client = OpenAI(api_key=openai_config["api_key"])
    
    max_retries = Settings.MAX_QUERY_RETRIES if Settings.ENABLE_QUERY_VALIDATION else 0
    
    # STEP 1: Reasoning and Intent Analysis
    reasoning_prompt = f"""You are an expert query analyzer. Analyze this question and extract structured information.

QUESTION: {natural_language_query}

Your task: Extract the following in JSON format:
{{
  "intent": "What is the user asking for? (e.g., 'find shared neighbors', 'count voxels', 'get properties')",
  "entities": ["List specific voxel IDs, types, or sensor properties mentioned"],
  "conditions": ["List any conditions like coordinate ranges, sensor filters, etc."],
  "relationship_pattern": "What graph pattern is needed? (e.g., 'two voxels and their shared neighbors', 'single voxel properties', 'path between voxels')",
  "specific_voxels": ["If question mentions 'these voxels', 'those', 'them', extract the IDs from context. If not clear, return empty list"],
  "reasoning_steps": [
    "Step 1: Understand what...",
    "Step 2: Identify that...",
    "Step 3: Determine..."
  ]
}}

CRITICAL: If the question refers to "these voxels", "those", "them", "it" - the entities should be extracted from the conversation context that's included in the query. Look for patterns like "voxel 5", "voxel ID 6", etc.

Return ONLY the JSON, no other text."""

    try:
        # STEP 1: Get reasoning analysis
        reasoning_response = client.chat.completions.create(
            model=openai_config["model"],
            messages=[
                {"role": "system", "content": "You are an expert at analyzing questions and extracting structured information."},
                {"role": "user", "content": reasoning_prompt}
            ],
            temperature=0.0
        )
        
        reasoning_text = reasoning_response.choices[0].message.content.strip()
        
        # Clean up JSON if wrapped in markdown
        if reasoning_text.startswith("```"):
            lines = reasoning_text.split("\n")
            reasoning_text = "\n".join([l for l in lines if not l.startswith("```")])
            reasoning_text = reasoning_text.strip()
        
        try:
            reasoning = json.loads(reasoning_text)
            print(f"\n🧠 REASONING ANALYSIS:")
            print(f"   Intent: {reasoning.get('intent', 'Unknown')}")
            print(f"   Entities: {reasoning.get('entities', [])}")
            print(f"   Specific Voxels: {reasoning.get('specific_voxels', [])}")
            print(f"   Pattern: {reasoning.get('relationship_pattern', 'Unknown')}")
            print(f"\n📋 Reasoning Steps:")
            for i, step in enumerate(reasoning.get('reasoning_steps', []), 1):
                print(f"   {i}. {step}")
        except json.JSONDecodeError:
            reasoning = {"error": "Failed to parse reasoning", "raw": reasoning_text}
            print(f"\n⚠️  Reasoning parse error, proceeding with basic analysis")
        
        # STEP 2: Generate Cypher with reasoning context
        cypher_prompt = f"""You are a Cypher query expert for Neo4j.

{DATABASE_SCHEMA}

QUESTION ANALYSIS (use this to generate accurate query):
{json.dumps(reasoning, indent=2)}

ORIGINAL QUESTION: {natural_language_query}

CRITICAL QUERY GENERATION RULES:
1. Always return specific properties (v.id, v.x, etc.), NEVER "RETURN v"
2. Use LIMIT to avoid too many results (default: 10)
3. ALWAYS use DISTINCT when returning related nodes to avoid duplicates
4. If specific voxel IDs are mentioned, use them in the query (e.g., WHERE v.id IN [5, 6])
5. For "shared neighbors" between TWO voxels, use this pattern:
   MATCH (v1:Voxel {{id: ID1}})-[:ADJACENT_TO]-(shared:Voxel)-[:ADJACENT_TO]-(v2:Voxel {{id: ID2}})
   WHERE v1.id <> v2.id
   RETURN DISTINCT shared.id, shared.x, shared.y, shared.z, shared.type
6. If the question mentions "these/those/them" and specific IDs are in the analysis, USE THEM!

7. SENSOR CHART NAMES vs DATABASE: If the user says **SG1** or **SG2** (strain gauges), map to
   Neo4j `sensor_id` **'S3'** (SG1) and **'S4'** (SG2). Never compare with exact `=` on floats —
   use `abs(a-b) < 2.0` (μE) for "same value", or detect crossing with consecutive differences
   `d0 * d1 < 0` as in DATABASE_SCHEMA examples 37–38.

8. FEM CYCLES / "how many simulations": There is only ONE `(:FEMAnalysis {analysis_id:'latest'})` node.
   It is reused every run. **Never** answer cycle count with `count(FEMAnalysis)` or `count(DISTINCT a)`.
   Always use: `MATCH (a:FEMAnalysis {analysis_id:'latest'}) RETURN coalesce(a.fem_cycle_count,0)`.

9. STRESS TIME PERIODS: Voxel `fem_timestamp` is overwritten each FEM cycle (all solids share the latest time).
   For a **list of when** stress was computed across many runs, use `a.fem_cycle_timestamps` on FEMAnalysis.

10. SENSOR VOCABULARY: "Voxels connected to sensors" means `WHERE v.sensor_id IS NOT NULL` on **Voxel**.
    Do NOT use non-existent patterns like `(Voxel)-[:CONNECTED_TO]->(Sensor)` or `(:Sensor)`.

🚨 CRITICAL SYNTAX RULES - AVOID COMMON ERRORS:
1. ❌ NEVER use multiple MATCH statements in one query - combine them!
   WRONG:
   MATCH (v:Voxel) RETURN v.id
   MATCH (a:FEMAnalysis) RETURN a.timestamp
   
   RIGHT:
   MATCH (v:Voxel), (a:FEMAnalysis)
   RETURN v.id, a.timestamp
   LIMIT 10

2. ❌ NEVER lose variable scope in WITH clauses
   WRONG:
   MATCH (v:Voxel)
   WITH v.stress_magnitude as stress
   RETURN count(v)  ← v is not defined!
   
   RIGHT:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL
   RETURN count(v), avg(v.stress_magnitude)

3. ✅ For comprehensive statistics, use a single aggregation:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL
   RETURN count(v) as total, avg(v.stress_magnitude) as avg_stress, max(v.stress_magnitude) as max_stress

4. ✅ For safety assessment, query ALL stats in ONE query:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL
   RETURN 
     count(v) as total_voxels,
     avg(v.stress_magnitude) as avg_stress,
     min(v.stress_magnitude) as min_stress,
     max(v.stress_magnitude) as max_stress,
     sum(CASE WHEN v.stress_magnitude > 3000 THEN 1 ELSE 0 END) as critical_count

OUTPUT: Return ONLY the Cypher query, no explanations, no markdown."""

        # STEP 3: Generate Cypher query
        response = client.chat.completions.create(
            model=openai_config["model"],
            messages=[
                {"role": "system", "content": cypher_prompt},
                {"role": "user", "content": "Generate the Cypher query now."}
            ],
            temperature=0.0
        )
        
        cypher_query = response.choices[0].message.content.strip()
        
        # Clean up the query
        if cypher_query.startswith("```"):
            lines = cypher_query.split("\n")
            cypher_query = "\n".join([l for l in lines if not l.startswith("```")])
            cypher_query = cypher_query.strip()
        
        print(f"\n🔍 Generated Cypher:\n   {cypher_query}")
        
        # STEP 4: Execute query with retry logic
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                results = _execute_query(cypher_query)
                
                print(f"✅ Query executed: {len(results)} results")
                
                if not results:
                    # Query ran fine but returned no rows.
                    # Try a safe fallback first before diagnosing.
                    fallback = _try_safe_fallback(natural_language_query)
                    if fallback is not None and len(fallback) > 0:
                        return json.dumps({
                            "success": True,
                            "results": fallback,
                            "count": len(fallback),
                            "note": "Answered using a simplified pre-validated query",
                            "attempts": attempt + 1,
                        }, indent=2, default=str)
                    diag = _auto_diagnose()
                    return json.dumps({
                        "query_status": "empty_results",
                        "results": [],
                        "available_summary": diag,
                        "hint": (
                            "The query ran but matched no rows.  "
                            "Use 'available_summary' to answer from aggregate data.  "
                            "Do NOT say 'there was an issue' or 'no results found' — "
                            "instead present what IS available from available_summary."
                        ),
                        "attempts": attempt + 1,
                    }, indent=2, default=str)
                
                return json.dumps({
                    "success": True,
                    "query": cypher_query,
                    "reasoning": reasoning,
                    "results": results,
                    "count": len(results),
                    "attempts": attempt + 1
                }, indent=2)
                
            except Exception as query_error:
                last_error = str(query_error)
                error_msg  = str(query_error)
                print(f"⚠️  Query error (attempt {attempt+1}): {error_msg[:120]}")

                # Retry on ANY error — not just syntax errors.
                # The LLM occasionally produces ordering issues (ORDER BY before
                # RETURN), bad property names, or scope errors that a second
                # LLM pass can fix.
                if attempt < max_retries:
                    print(f"🔄 Attempt {attempt+1}/{max_retries+1} failed — asking LLM to fix…")
                    cypher_query = _validate_and_fix_query(
                        cypher_query, error_msg, reasoning,
                        natural_language_query, client, openai_config
                    )
                    print(f"🔍 Retry with fixed query:\n   {cypher_query}")
                    continue
                else:
                    print(f"❌ Max retries ({max_retries}) reached")
                    break

        # ── All LLM-generated attempts failed ────────────────────────────────
        # Before giving up, try a pre-validated safe fallback for known intents.
        print("🔄 Trying safe fallback query…")
        fallback_results = _try_safe_fallback(natural_language_query)

        if fallback_results is not None and len(fallback_results) > 0:
            # Fallback worked — return its results as a normal success
            return json.dumps({
                "success": True,
                "results": fallback_results,
                "count": len(fallback_results),
                "note": "Answered using a simplified pre-validated query",
                "attempts": max_retries + 1,
            }, indent=2, default=str)

        # ── Still nothing — return available summary data without error framing ──
        diag = _auto_diagnose()
        return json.dumps({
            # IMPORTANT: do NOT use "success": False — that phrase triggers the
            # agent to say "there was an issue".  Use a neutral status key.
            "query_status": "no_row_results",
            "results": [],
            "available_summary": diag,
            "hint": (
                "The row-level query could not execute.  "
                "Use 'available_summary' to answer from aggregate data "
                "without mentioning query errors to the user.  "
                "Present the summary as your primary answer."
            ),
        }, indent=2, default=str)

    except Exception as e:
        diag = _auto_diagnose()
        return json.dumps({
            "query_status": "generation_error",
            "results": [],
            "available_summary": diag,
            "hint": (
                "Query generation encountered an error.  "
                "Answer from 'available_summary' without mentioning errors."
            ),
        }, indent=2, default=str)


def get_database_schema() -> str:
    """
    Get the Neo4j database schema and statistics.
    
    Use this to understand what data is available before querying.
    
    Returns:
        Database schema and statistics as JSON string
    """
    try:
        # Get node counts
        node_counts = _execute_query("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
        
        # Get relationship counts
        rel_counts = _execute_query("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
        
        # Get sample voxel to show properties
        sample = _execute_query("MATCH (v:Voxel) RETURN v.id, v.x, v.y, v.z, v.type, v.connection_count, v.ground_connected, v.sensor_strain_uE, v.stress_magnitude LIMIT 1")
        
        # Get voxel types
        types = _execute_query("MATCH (v:Voxel) RETURN DISTINCT v.type as type")
        
        # Count voxels with sensor data
        sensor_count = _execute_query("MATCH (v:Voxel) WHERE v.sensor_strain_uE IS NOT NULL RETURN count(v) as count")
        
        schema_info = {
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
            "sample_voxel": sample[0] if sample else None,
            "available_types": [t["type"] for t in types],
            "voxels_with_sensor_data": sensor_count[0]["count"] if sensor_count else 0,
            "schema": DATABASE_SCHEMA
        }
        
        return json.dumps(schema_info, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def get_property_history(voxel_id: int, property_name: Optional[str] = None) -> str:
    """
    Get the version history of a voxel's properties.
    
    Shows all historical values with timestamps in format:
    temp_c[0]: "25.5" at 2025-10-12T10:00:00
    temp_c[1]: "26.0" at 2025-10-12T10:32:15
    
    Args:
        voxel_id: Voxel ID to get history for
        property_name: Optional specific property (e.g. "temp_c", "strain_uE", "load_N", "type")
                      If None, returns history for all properties
        
    Returns:
        JSON string with complete version history
    """
    try:
        print(f"\n📜 Fetching property history for voxel {voxel_id}...")
        
        if property_name:
            # Get history for specific property
            query = """
                MATCH (vp:VoxelProperty {voxel_id: $voxel_id, property_name: $property_name})
                RETURN vp.property_name as property_name,
                       vp.property_value as value,
                       vp.version_number as version,
                       vp.timestamp as timestamp,
                       vp.change_type as change_type
                ORDER BY vp.version_number ASC
            """
            results = _execute_query(query, {"voxel_id": voxel_id, "property_name": property_name})
        else:
            # Get history for all properties
            query = """
                MATCH (vp:VoxelProperty {voxel_id: $voxel_id})
                RETURN vp.property_name as property_name,
                       vp.property_value as value,
                       vp.version_number as version,
                       vp.timestamp as timestamp,
                       vp.change_type as change_type
                ORDER BY vp.property_name ASC, vp.version_number ASC
            """
            results = _execute_query(query, {"voxel_id": voxel_id})
        
        if not results:
            return json.dumps({
                "status": "no_history",
                "voxel_id": voxel_id,
                "message": "No version history found. Voxel may not exist or has no tracked properties."
            }, indent=2)
        
        # Organize by property name
        history_by_property = {}
        for record in results:
            prop_name = record["property_name"]
            if prop_name not in history_by_property:
                history_by_property[prop_name] = []
            
            history_by_property[prop_name].append({
                "version": record["version"],
                "value": record["value"],
                "timestamp": record["timestamp"],
                "change_type": record.get("change_type", "unknown")
            })
        
        # Build human-readable summary
        summary_lines = []
        for prop_name, versions in history_by_property.items():
            summary_lines.append(f"\n{prop_name.upper()} HISTORY:")
            for v in versions:
                version_num = v["version"]
                value = v["value"]
                timestamp = v["timestamp"]
                change_type = v["change_type"]
                summary_lines.append(f"  [{version_num}] = \"{value}\" at {timestamp} ({change_type})")
        
        current_values = {}
        for prop_name, versions in history_by_property.items():
            latest = versions[-1]  # Last version is current
            current_values[prop_name] = latest["value"]
        
        result = {
            "status": "success",
            "voxel_id": voxel_id,
            "property_filter": property_name if property_name else "all",
            "current_values": current_values,
            "version_history": history_by_property,
            "summary": "\n".join(summary_lines),
            "total_versions": sum(len(versions) for versions in history_by_property.values())
        }
        
        print(f"✅ Found {result['total_versions']} version entries")
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "message": "Failed to retrieve property history"
        }, indent=2)


def check_recent_updates(minutes: int = 5, voxel_ids: Optional[List[int]] = None) -> str:
    """
    Check for recent updates/changes to voxels in the database.
    
    CRITICAL: ALWAYS call this BEFORE answering questions about voxels!
    This ensures you have the latest information about any changes.
    
    Args:
        minutes: Look for changes in the last N minutes (default: 5)
        voxel_ids: Optional list of specific voxel IDs to check (if None, checks all)
        
    Returns:
        JSON string with recent changes information
    """
    try:
        from datetime import datetime, timedelta
        
        # Calculate cutoff time
        cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        print(f"\n🔍 Checking for updates in the last {minutes} minutes...")
        
        # Check for ChangeNotifications (if they exist)
        change_notifications = []
        try:
            if voxel_ids:
                # Check specific voxels
                query = """
                    MATCH (n:ChangeNotification)
                    WHERE n.timestamp > $cutoff AND n.voxel_id IN $voxel_ids
                    RETURN n.voxel_id as voxel_id, n.change_type as change_type,
                           n.timestamp as timestamp, 
                           COALESCE(n.old_value, 'null') as old_value,
                           COALESCE(n.new_value, 'null') as new_value
                    ORDER BY n.timestamp DESC
                    LIMIT 20
                """
                change_notifications = _execute_query(query, {"cutoff": cutoff, "voxel_ids": voxel_ids})
            else:
                # Check all recent changes
                query = """
                    MATCH (n:ChangeNotification)
                    WHERE n.timestamp > $cutoff
                    RETURN n.voxel_id as voxel_id, n.change_type as change_type,
                           n.timestamp as timestamp,
                           COALESCE(n.old_value, 'null') as old_value,
                           COALESCE(n.new_value, 'null') as new_value
                    ORDER BY n.timestamp DESC
                    LIMIT 20
                """
                change_notifications = _execute_query(query, {"cutoff": cutoff})
        except Exception:
            pass  # ChangeNotification nodes may not exist yet
        
        # Check for recently updated voxels
        recently_updated = []
        try:
            if voxel_ids:
                query = """
                    MATCH (v:Voxel)
                    WHERE v.last_updated > $cutoff AND v.id IN $voxel_ids
                    RETURN v.id as id, v.type as type,
                           v.x as x, v.y as y, v.z as z, v.last_updated as last_updated,
                           v.sensor_strain_uE as sensor_strain_uE,
                           v.stress_magnitude as stress_magnitude
                    ORDER BY v.last_updated DESC
                """
                recently_updated = _execute_query(query, {"cutoff": cutoff, "voxel_ids": voxel_ids})
            else:
                query = """
                    MATCH (v:Voxel)
                    WHERE v.last_updated > $cutoff
                    RETURN v.id as id, v.type as type,
                           v.x as x, v.y as y, v.z as z, v.last_updated as last_updated,
                           v.sensor_strain_uE as sensor_strain_uE,
                           v.stress_magnitude as stress_magnitude
                    ORDER BY v.last_updated DESC
                    LIMIT 20
                """
                recently_updated = _execute_query(query, {"cutoff": cutoff})
        except Exception:
            pass  # last_updated may not exist on voxels

        # Check for proactive structural alerts
        structural_alerts = []
        try:
            if voxel_ids:
                alert_query = """
                    MATCH (a:StructuralAlert)
                    WHERE a.timestamp > $cutoff AND a.voxel_id IN $voxel_ids
                    RETURN a.alert_id as alert_id,
                           a.alert_type as alert_type,
                           a.severity as severity,
                           a.timestamp as timestamp,
                           a.voxel_id as voxel_id,
                           a.message as message,
                           a.value as value
                    ORDER BY a.timestamp DESC
                    LIMIT 20
                """
                structural_alerts = _execute_query(alert_query, {"cutoff": cutoff, "voxel_ids": voxel_ids})
            else:
                alert_query = """
                    MATCH (a:StructuralAlert)
                    WHERE a.timestamp > $cutoff
                    RETURN a.alert_id as alert_id,
                           a.alert_type as alert_type,
                           a.severity as severity,
                           a.timestamp as timestamp,
                           a.voxel_id as voxel_id,
                           a.message as message,
                           a.value as value
                    ORDER BY a.timestamp DESC
                    LIMIT 20
                """
                structural_alerts = _execute_query(alert_query, {"cutoff": cutoff})
        except Exception:
            pass
        
        # Build detailed summary for agent to analyze
        change_summary = []
        
        # Process change notifications with full details
        for change in change_notifications:
            voxel_id = change.get("voxel_id")
            change_type = change.get("change_type")
            timestamp = change.get("timestamp", "unknown")
            old_value = change.get("old_value")
            new_value = change.get("new_value")
            
            # Build description based on change type
            if change_type.endswith("_update"):
                prop_name = change_type.replace("_update", "")
                # Handle null values explicitly
                old_str = "null" if old_value in ['None', 'null', None] else str(old_value)
                new_str = "null" if new_value in ['None', 'null', None] else str(new_value)
                
                # Format with previous/current labels
                description = f"Voxel {voxel_id}: {prop_name} - previous: {old_str}, current: {new_str}"
            else:
                old_str = "null" if old_value in ['None', 'null', None] else str(old_value)
                new_str = "null" if new_value in ['None', 'null', None] else str(new_value)
                description = f"Voxel {voxel_id}: {change_type} - previous: {old_str}, current: {new_str}"
            
            detail = {
                "voxel_id": voxel_id,
                "change_type": change_type,
                "timestamp": timestamp,
                "old_value": old_value,
                "new_value": new_value,
                "description": description
            }
            change_summary.append(detail)
        
        # Process recently updated voxels with full details (only if not already in change_summary)
        processed_voxels = set(c.get("voxel_id") for c in change_summary)
        
        for voxel in recently_updated:
            voxel_id = voxel.get("id")
            # Skip if already processed in change notifications
            if voxel_id in processed_voxels:
                continue
                
            voxel_type = voxel.get("type")
            x, y, z = voxel.get("x"), voxel.get("y"), voxel.get("z")
            last_updated = voxel.get("last_updated")
            temp_c = voxel.get("temp_c")
            strain_uE = voxel.get("strain_uE")
            load_N = voxel.get("load_N")
            
            # Only mention sensor properties that have values
            sensor_info = []
            if temp_c is not None:
                sensor_info.append(f"temp={temp_c}°C")
            if strain_uE is not None:
                sensor_info.append(f"strain={strain_uE}μE")
            if load_N is not None:
                sensor_info.append(f"load={load_N}N")
            
            sensor_text = ", ".join(sensor_info) if sensor_info else "no sensor data"
            
            detail = {
                "voxel_id": voxel_id,
                "current_state": {
                    "type": voxel_type,
                    "position": {"x": x, "y": y, "z": z},
                    "temp_c": temp_c,
                    "strain_uE": strain_uE,
                    "load_N": load_N
                },
                "last_updated": last_updated,
                "description": f"Voxel {voxel_id} was updated - position=({x:.2f}, {y:.2f}, {z:.2f}), {sensor_text}"
            }
            change_summary.append(detail)

        # Include proactive monitoring alerts in response context
        for alert in structural_alerts:
            detail = {
                "alert_id": alert.get("alert_id"),
                "alert_type": alert.get("alert_type"),
                "severity": alert.get("severity"),
                "timestamp": alert.get("timestamp"),
                "voxel_id": alert.get("voxel_id"),
                "value": alert.get("value"),
                "description": alert.get("message", "Structural alert raised")
            }
            change_summary.append(detail)
        
        result = {
            "status": "checked",
            "cutoff_time": cutoff,
            "minutes_checked": minutes,
            "checked_voxels": voxel_ids if voxel_ids else "all",
            "has_changes": len(change_notifications) > 0 or len(recently_updated) > 0 or len(structural_alerts) > 0,
            "total_changes": len(change_notifications) + len(recently_updated) + len(structural_alerts),
            "change_details": change_summary,
            "raw_change_notifications": change_notifications,
            "raw_recently_updated_voxels": recently_updated,
            "raw_structural_alerts": structural_alerts
        }
        
        if result["has_changes"]:
            print(f"⚠️  Found {result['total_changes']} recent changes!")
            print(f"📋 Change details available for agent analysis")
        else:
            print(f"✅ No recent changes detected")
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "message": "Failed to check for updates"
        }, indent=2)


def detect_structural_events(
    shake_g_threshold: float = 0.15,
    strain_spike_threshold_uE: float = 30.0,
    history_window: int = 5,
) -> str:
    """
    Dedicated structural-event detector.  Always call this when the user asks about:
    shake, vibration, impact, movement, anomaly, structural change, spike, anything felt.

    Runs four independent checks against live Neo4j sensor data:
      1. Acceleration magnitude spike (shake/impact) on MPU sensor voxels
      2. Net acceleration vector deviation from baseline (1g in Z direction)
      3. Strain spike on SG sensor voxels
      4. Fallback: whether sensor data exists at all

    Args:
        shake_g_threshold:        deviation in g-force that counts as a shake (default 0.15g)
        strain_spike_threshold_uE: μE deviation from rolling average that counts as a spike (default 30)
        history_window:           number of recent readings to use for rolling average (default 5)

    Returns:
        JSON string with findings, anomalies, timestamps, and a plain-English summary.
    """
    results = {
        "physical_sensors":    {},   # sensor_id → {type, voxel_count, latest_readings}
        "shake_events":        [],   # MPU-only
        "strain_spikes":       [],   # SG-only
        "sensor_count":        0,    # unique physical sensor IDs (should be 4)
        "history_available":   False,
        "summary":             "",
    }

    try:
        # ── 1. Count UNIQUE physical sensors by sensor_id ─────────────────
        # SG sensors cover many strut voxels — we count distinct sensor_id values
        sensor_id_q = _execute_query("""
            MATCH (v:Voxel)
            WHERE v.sensor_id IS NOT NULL
            RETURN v.sensor_id AS sid, v.sensor_type AS stype, count(v) AS voxel_count
            ORDER BY v.sensor_id
        """)
        for row in sensor_id_q:
            sid = row["sid"]
            results["physical_sensors"][sid] = {
                "type":        row["stype"],
                "voxel_count": row["voxel_count"],
            }
        results["sensor_count"] = len(results["physical_sensors"])

        # If no tagged voxels yet, fall back to old-style count
        if results["sensor_count"] == 0:
            old_count_q = _execute_query(
                "MATCH (v:Voxel) WHERE v.sensor_strain_uE IS NOT NULL RETURN count(v) AS n"
            )
            old_n = old_count_q[0]["n"] if old_count_q else 0
            if old_n == 0:
                results["summary"] = (
                    "No sensor data in Neo4j yet. Sensors may not be assigned or the MQTT "
                    "pipeline hasn't written data. The UI charts show live WebSocket data "
                    "which bypasses Neo4j storage."
                )
                return json.dumps(results, indent=2)
            else:
                results["summary"] = (
                    f"{old_n} voxels have sensor_strain_uE data but are not yet tagged "
                    "with sensor_id/sensor_type. Restart the server so the latest "
                    "sensor_update.py code tags them correctly."
                )
                return json.dumps(results, indent=2)

        # ── 2. MPU sensors: shake detection from acc scalars ──────────────
        mpu_q = _execute_query("""
            MATCH (v:Voxel)
            WHERE v.sensor_type = 'MPU'
            RETURN v.id AS id, v.sensor_id AS sid,
                   v.grid_i AS gi, v.grid_j AS gj, v.grid_k AS gk,
                   v.x AS x, v.y AS y, v.z AS z,
                   v.sensor_acc_x  AS ax, v.sensor_acc_y AS ay, v.sensor_acc_z AS az,
                   v.sensor_gyro_x AS gx, v.sensor_gyro_y AS gy, v.sensor_gyro_z AS gz,
                   v.sensor_strain_uE AS strain,
                   v.last_updated AS ts
        """)

        for r in mpu_q:
            ax = r.get("ax") or 0.0
            ay = r.get("ay") or 0.0
            az = r.get("az") or 1.0  # default resting
            mag = (ax**2 + ay**2 + az**2) ** 0.5
            deviation = abs(mag - 1.0)
            sid = r.get("sid", "?")
            # Store latest in physical_sensors dict
            results["physical_sensors"].setdefault(sid, {})["latest_acc"] = {
                "x": round(ax, 4), "y": round(ay, 4), "z": round(az, 4),
                "magnitude": round(mag, 4), "deviation_from_1g": round(deviation, 4),
            }
            if deviation > shake_g_threshold:
                results["shake_events"].append({
                    "sensor_id":        sid,
                    "voxel_id":         r.get("id"),
                    "pos":              [round(r.get("x", 0), 3),
                                         round(r.get("y", 0), 3),
                                         round(r.get("z", 0), 3)],
                    "acc":              {"x": round(ax, 4), "y": round(ay, 4), "z": round(az, 4)},
                    "magnitude_g":      round(mag, 4),
                    "deviation_from_1g":round(deviation, 4),
                    "gyro_dps":         {"x": round(r.get("gx") or 0, 2),
                                         "y": round(r.get("gy") or 0, 2),
                                         "z": round(r.get("gz") or 0, 2)},
                    "timestamp":        r.get("ts"),
                })

        # ── 3. MPU history: acc-z spike detection ─────────────────────────
        mpu_hist_q = _execute_query("""
            MATCH (v:Voxel)
            WHERE v.sensor_type = 'MPU'
              AND v.sensor_acc_z_history IS NOT NULL
              AND size(v.sensor_acc_z_history) >= 3
            RETURN v.id AS id, v.sensor_id AS sid,
                   v.sensor_acc_z_history     AS azh,
                   v.sensor_timestamp_history AS tsh
        """)
        if mpu_hist_q:
            results["history_available"] = True
        for r in mpu_hist_q:
            azh = r.get("azh") or []
            tsh = r.get("tsh") or []
            if len(azh) < 3:
                continue
            latest_az = azh[-1]
            window_az = azh[-history_window:] if len(azh) >= history_window else azh[:-1]
            avg_az    = sum(window_az) / len(window_az) if window_az else 1.0
            dev       = abs(latest_az - 1.0)
            if dev > shake_g_threshold:
                sid = r.get("sid", "?")
                already = any(e["sensor_id"] == sid for e in results["shake_events"])
                if not already:
                    results["shake_events"].append({
                        "sensor_id":     sid,
                        "voxel_id":      r.get("id"),
                        "source":        "acc_z_history",
                        "latest_acc_z":  round(latest_az, 4),
                        "avg_acc_z":     round(avg_az, 4),
                        "deviation":     round(dev, 4),
                        "timestamp":     tsh[-1] if tsh else None,
                    })

        # ── 4. SG sensors: strain spike detection from history ─────────────
        sg_hist_q = _execute_query("""
            MATCH (v:Voxel)
            WHERE v.sensor_type = 'SG'
              AND v.sensor_strain_history IS NOT NULL
              AND size(v.sensor_strain_history) >= 3
            RETURN v.sensor_id AS sid,
                   avg(v.sensor_strain_uE) AS latest_avg_strain,
                   collect(v.sensor_strain_history) AS all_histories,
                   collect(v.sensor_timestamp_history) AS all_timestamps
        """)
        for r in sg_hist_q:
            # For strut sensors: average across all strut voxels' histories
            all_h = r.get("all_histories") or []
            all_t = r.get("all_timestamps") or []
            if not all_h:
                continue
            # Flatten and align by position
            min_len = min(len(h) for h in all_h if h)
            if min_len < 3:
                continue
            # Average strain per reading position across strut voxels
            avg_per_step = [
                sum(h[i] for h in all_h if h and i < len(h)) / len(all_h)
                for i in range(min_len)
            ]
            latest_s = avg_per_step[-1]
            window   = avg_per_step[-history_window:] if len(avg_per_step) >= history_window else avg_per_step[:-1]
            avg_s    = sum(window) / len(window) if window else 0.0
            delta_s  = latest_s - avg_s
            sid = r.get("sid", "?")
            ts  = all_t[0][-1] if all_t and all_t[0] else None
            results["physical_sensors"].setdefault(sid, {})["latest_avg_strain"] = round(latest_s, 2)
            if abs(delta_s) > strain_spike_threshold_uE:
                results["strain_spikes"].append({
                    "sensor_id":      sid,
                    "latest_strain":  round(latest_s, 2),
                    "rolling_avg":    round(avg_s, 2),
                    "delta_uE":       round(delta_s, 2),
                    "reading_count":  min_len,
                    "timestamp":      ts,
                })

        # ── 5. SG scalar fallback (if no history yet) ─────────────────────
        if not sg_hist_q:
            sg_scalar_q = _execute_query("""
                MATCH (v:Voxel)
                WHERE v.sensor_type = 'SG'
                RETURN v.sensor_id AS sid, avg(v.sensor_strain_uE) AS avg_strain,
                       v.last_updated AS ts
            """)
            for r in sg_scalar_q:
                sid = r.get("sid", "?")
                results["physical_sensors"].setdefault(sid, {})["latest_avg_strain"] = (
                    round(r.get("avg_strain") or 0, 2)
                )

        # ── 6. Plain-English summary ───────────────────────────────────────
        _sens_parts = ", ".join(
            f"{k}({v.get('type', '?')})" for k, v in sorted(results["physical_sensors"].items())
        )
        lines = [f"Physical sensors in Neo4j: {results['sensor_count']} ({_sens_parts})"]

        if results["shake_events"]:
            lines.append(f"\n⚠️  SHAKE / VIBRATION DETECTED on {len(results['shake_events'])} MPU sensor(s):")
            for ev in results["shake_events"]:
                mag = ev.get("magnitude_g") or ev.get("deviation", "?")
                dev = ev.get("deviation_from_1g") or ev.get("deviation", "?")
                lines.append(
                    f"  • {ev.get('sensor_id','?')}: acc magnitude={mag}g, "
                    f"deviation from 1g={dev}g  [{ev.get('timestamp','?')}]"
                )
        else:
            lines.append(f"\nNo MPU shake detected (threshold: >{shake_g_threshold}g deviation from 1g).")

        if results["strain_spikes"]:
            lines.append(f"\n⚠️  STRAIN SPIKE on {len(results['strain_spikes'])} SG sensor(s):")
            for sp in results["strain_spikes"]:
                lines.append(
                    f"  • {sp['sensor_id']}: jumped {sp['delta_uE']:+.1f}μE "
                    f"(latest={sp['latest_strain']}μE vs avg={sp['rolling_avg']}μE)"
                    f"  [{sp.get('timestamp','?')}]"
                )
        else:
            if results["history_available"]:
                lines.append(f"No SG strain spikes above {strain_spike_threshold_uE}μE.")
            else:
                lines.append("SG strain history not yet built (need ≥3 MQTT cycles).")

        results["summary"] = "\n".join(lines)

    except Exception as e:
        results["summary"] = f"Detection error: {e}"
        results["error"] = str(e)

    return json.dumps(results, indent=2, default=str)


def reset_sensor_and_fem_data() -> Dict[str, Any]:
    """
    Wipe all sensor readings and FEM results from the graph while keeping
    the voxel structure (grid coordinates, positions, neighbour relationships)
    intact.  Call this every time MQTT (re)connects so the new recording
    session starts with a clean slate.

    Clears:
      • All sensor_* properties on Voxel nodes (including history arrays)
      • All FEM scalar / stress / strain properties on Voxel nodes
      • The FEMAnalysis node(s) (deleted entirely)
      • The SensorStream node(s) (deleted entirely)
    """
    results: Dict[str, Any] = {}
    try:
        conn_info = Settings.get_connection_info()
        db = conn_info.get("database")
        driver = _get_driver()
        ctx = driver.session(database=db) if db else driver.session()

        with ctx as session:
            # ── 1. Remove sensor properties from voxels ─────────────────────
            r1 = session.run("""
                MATCH (v:Voxel)
                WHERE v.sensor_id IS NOT NULL
                   OR v.sensor_strain_uE IS NOT NULL
                REMOVE
                    v.sensor_id,
                    v.sensor_type,
                    v.sensor_strain_uE,
                    v.sensor_hx711_raw,
                    v.sensor_acc_x,    v.sensor_acc_y,    v.sensor_acc_z,
                    v.sensor_gyro_x,   v.sensor_gyro_y,   v.sensor_gyro_z,
                    v.sensor_strain_history,
                    v.sensor_acc_x_history, v.sensor_acc_y_history, v.sensor_acc_z_history,
                    v.sensor_gyro_x_history, v.sensor_gyro_y_history, v.sensor_gyro_z_history,
                    v.sensor_timestamp_history,
                    v.last_updated
                RETURN count(v) AS cleared
            """)
            results["sensor_voxels_cleared"] = (r1.single() or {}).get("cleared", 0)

            # ── 2. Remove FEM properties from voxels ─────────────────────────
            r2 = session.run("""
                MATCH (v:Voxel)
                WHERE v.stress_magnitude IS NOT NULL
                   OR v.eps_xx IS NOT NULL
                REMOVE
                    v.eps_xx, v.eps_yy, v.eps_zz,
                    v.sigma_xx, v.sigma_yy, v.sigma_zz,
                    v.sigma_xy, v.sigma_yz, v.sigma_xz,
                    v.stress_magnitude,
                    v.fem_timestamp
                RETURN count(v) AS cleared
            """)
            results["fem_voxels_cleared"] = (r2.single() or {}).get("cleared", 0)

            # ── 3. Delete FEMAnalysis node(s) ────────────────────────────────
            r3 = session.run("""
                MATCH (a:FEMAnalysis)
                WITH a, a.analysis_id AS aid
                DETACH DELETE a
                RETURN count(aid) AS deleted
            """)
            results["fem_analyses_deleted"] = (r3.single() or {}).get("deleted", 0)

            # ── 4. Delete SensorStream node(s) ───────────────────────────────
            r4 = session.run("""
                MATCH (s:SensorStream)
                WITH s, s.id AS sid
                DETACH DELETE s
                RETURN count(sid) AS deleted
            """)
            results["sensor_streams_deleted"] = (r4.single() or {}).get("deleted", 0)

        results["status"] = "ok"

    except Exception as exc:
        results["status"] = "error"
        results["error"] = str(exc)

    return results


# Create Agno Toolkit for Neo4j (intelligent query generation + change awareness + version history)
neo4j_toolkit = Toolkit(
    name="neo4j_tools",
    tools=[
        probe_database_state,     # ← Diagnostic: call first when results are empty/unknown
        detect_structural_events, # ← Use for shake/vibration/anomaly questions
        check_recent_updates,     # Check for recent data changes
        get_property_history,     # Get version history of properties
        intelligent_query_neo4j,  # Main intelligent Cypher tool
        get_database_schema       # Full schema helper
    ]
)


# Export for easy import
__all__ = [
    "neo4j_toolkit", "intelligent_query_neo4j", "get_database_schema",
    "check_recent_updates", "get_property_history", "detect_structural_events",
    "probe_database_state", "reset_sensor_and_fem_data",
]

