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


def _execute_query(cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
    """Execute a Cypher query and convert Neo4j objects to dictionaries"""
    conn_info = Settings.get_connection_info()
    with _get_driver().session(database=conn_info["database"]) as session:
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
    
  Sensor Properties (may be null if no sensor data):
    - temp_c (float): Temperature in Celsius
    - strain_uE (float): Strain in microstrains (μE)
    - load_N (float): Load in Newtons
    - hx711_raw (int): Raw HX711 load cell reading
    - acc_g_x, acc_g_y, acc_g_z (float): Acceleration in g-force (x, y, z)
    - gyro_dps_x, gyro_dps_y, gyro_dps_z (float): Gyroscope in degrees/second (x, y, z)
    - quality_ok (boolean): Whether sensor readings are valid
    - quality_flags (string): Comma-separated quality issue flags
  
  FEM Analysis Properties (ARRAYS - versioned history, never deleted):
    ⚠️ CRITICAL: ALL FEM properties are ARRAYS storing version history!
    - stress_magnitude (array of floats): Von Mises stress magnitude in Pascals (Pa)
    - eps_xx, eps_yy, eps_zz (arrays of floats): Strain tensor components
    - sigma_xx, sigma_yy, sigma_zz (arrays of floats): Normal stress components in Pa
    - sigma_xy, sigma_yz, sigma_xz (arrays of floats): Shear stress components in Pa
    - fem_versions (array of ints): Version numbers for each FEM run
    - fem_timestamps (array of strings): ISO timestamps for each FEM run
    - last_fem_version (int): Most recent FEM version number
    - last_updated (string): Most recent update timestamp
    
    🔍 ARRAY ACCESS RULES:
    - Latest value: property[size(property)-1]  ← ALWAYS use this!
    - NEVER use property[-1] (not supported in Cypher)
    - First value: property[0]
    - Check array exists: WHERE property IS NOT NULL AND size(property) > 0
    - Array grows with each FEM run: coalesce(property, []) + [new_value]

- FEMAnalysis
  Properties:
    - analysis_id (string): Unique analysis identifier
    - timestamp (string): ISO timestamp of analysis
    - sensor_reading (string): Sensor data used for analysis
    - version (int): Analysis version number
    - total_voxels (int): Number of voxels analyzed
    - status (string): Analysis status ('completed', 'failed', etc.)

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

7. Find voxels with sensor data (temperature):
   MATCH (v:Voxel)
   WHERE v.temp_c IS NOT NULL
   RETURN v.id, v.temp_c, v.x, v.y, v.z
   LIMIT 10

8. Find voxels with high strain:
   MATCH (v:Voxel)
   WHERE v.strain_uE > 1000
   RETURN v.id, v.strain_uE, v.x, v.y, v.z, v.type
   ORDER BY v.strain_uE DESC
   LIMIT 10

9. Find voxels with quality issues:
   MATCH (v:Voxel)
   WHERE v.quality_ok = false
   RETURN v.id, v.quality_flags, v.x, v.y, v.z
   LIMIT 10

10. Find closest voxel to a point or another voxel:
   MATCH (target:Voxel {id: 5})
   MATCH (v:Voxel) WHERE v.id <> 5
   RETURN v.id, v.type, 
          point.distance(point({x: target.x, y: target.y, z: target.z}), 
                         point({x: v.x, y: v.y, z: v.z})) as distance
   ORDER BY distance
   LIMIT 1
   
   NOTE: Use point.distance() NOT distance() (deprecated in newer Neo4j versions)

FEM STRESS/STRAIN QUERIES (CRITICAL - Arrays!):

11. Get maximum stress value (latest version):
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   WITH v, v.stress_magnitude[size(v.stress_magnitude)-1] as current_stress
   ORDER BY current_stress DESC
   LIMIT 1
   RETURN v.grid_i, v.grid_j, v.grid_k, v.x, v.y, v.z, current_stress

12. Get average stress statistics:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   WITH v.stress_magnitude[size(v.stress_magnitude)-1] as stress
   RETURN avg(stress)/1000 as avg_kPa, 
          min(stress)/1000 as min_kPa, 
          max(stress)/1000 as max_kPa,
          count(v) as voxel_count

13. Find high stress regions (>500kPa):
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   WITH v, v.stress_magnitude[size(v.stress_magnitude)-1] as stress
   WHERE stress > 500000
   RETURN v.grid_i, v.grid_j, v.grid_k, v.x, v.y, v.z, stress/1000 as stress_kPa
   ORDER BY stress DESC

14. Get strain components for a voxel:
   MATCH (v:Voxel {grid_i: 10, grid_j: 20, grid_k: 30})
   WHERE v.eps_xx IS NOT NULL AND size(v.eps_xx) > 0
   RETURN v.grid_i, v.grid_j, v.grid_k,
          v.eps_xx[size(v.eps_xx)-1] as eps_xx,
          v.eps_yy[size(v.eps_yy)-1] as eps_yy,
          v.eps_zz[size(v.eps_zz)-1] as eps_zz

15. Get stress tensor components:
   MATCH (v:Voxel)
   WHERE v.sigma_xx IS NOT NULL AND size(v.sigma_xx) > 0
   WITH v,
        v.sigma_xx[size(v.sigma_xx)-1] as sxx,
        v.sigma_yy[size(v.sigma_yy)-1] as syy,
        v.sigma_zz[size(v.sigma_zz)-1] as szz
   RETURN v.grid_i, v.grid_j, v.grid_k, sxx/1000 as sxx_kPa, syy/1000 as syy_kPa, szz/1000 as szz_kPa
   LIMIT 10

16. Get stress history for a voxel:
   MATCH (v:Voxel {grid_i: 10, grid_j: 20, grid_k: 30})
   WHERE v.stress_magnitude IS NOT NULL
   RETURN v.grid_i, v.grid_j, v.grid_k,
          v.stress_magnitude as stress_history,
          v.fem_timestamps as timestamps,
          size(v.stress_magnitude) as version_count

17. Get all FEM analysis sessions:
   MATCH (a:FEMAnalysis)
   RETURN a.analysis_id, a.timestamp, a.version, a.total_voxels, a.status
   ORDER BY a.timestamp DESC
   LIMIT 10

18. Count voxels with FEM data:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   RETURN count(v) as voxels_with_fem_data

19. Stress distribution by category:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   WITH v.stress_magnitude[size(v.stress_magnitude)-1] as stress
   RETURN 
     CASE 
       WHEN stress < 100000 THEN 'Low (<100kPa)'
       WHEN stress < 500000 THEN 'Medium (100-500kPa)'
       WHEN stress < 1000000 THEN 'High (500-1000kPa)'
       ELSE 'Critical (>1000kPa)'
     END as category,
     count(*) as voxel_count
   ORDER BY voxel_count DESC

SAFETY ASSESSMENT QUERIES (Multi-step reasoning):

20. Comprehensive safety statistics:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   WITH v.stress_magnitude[size(v.stress_magnitude)-1] as stress
   WITH 
     count(v) as total_voxels,
     avg(stress) as avg_stress,
     min(stress) as min_stress,
     max(stress) as max_stress,
     sum(CASE WHEN stress > 3000 THEN 1 ELSE 0 END) as critical_count,
     sum(CASE WHEN stress > 2000 AND stress <= 3000 THEN 1 ELSE 0 END) as high_count,
     sum(CASE WHEN stress > 1000 AND stress <= 2000 THEN 1 ELSE 0 END) as medium_count
   RETURN 
     total_voxels,
     avg_stress, min_stress, max_stress,
     critical_count, high_count, medium_count,
     toFloat(critical_count) / total_voxels * 100 as critical_percentage

21. Identify critical danger zones:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   WITH v, v.stress_magnitude[size(v.stress_magnitude)-1] as stress
   WHERE stress > 3000
   RETURN v.grid_i, v.grid_j, v.grid_k, v.x, v.y, v.z, 
          stress as stress_Pa, v.type,
          v.last_updated as timestamp
   ORDER BY stress DESC
   LIMIT 20

22. Temporal trend analysis (stress increasing/decreasing):
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) >= 2
   WITH v,
        v.stress_magnitude[0] as first_stress,
        v.stress_magnitude[size(v.stress_magnitude)-1] as latest_stress,
        size(v.stress_magnitude) as version_count
   WITH v, first_stress, latest_stress, version_count,
        latest_stress - first_stress as stress_change,
        (latest_stress - first_stress) / first_stress * 100 as percent_change
   WHERE abs(percent_change) > 10
   RETURN v.grid_i, v.grid_j, v.grid_k, v.x, v.y, v.z,
          first_stress, latest_stress, stress_change, percent_change,
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
   WHERE v1.stress_magnitude IS NOT NULL AND size(v1.stress_magnitude) > 0
     AND v2.stress_magnitude IS NOT NULL AND size(v2.stress_magnitude) > 0
   WITH v1, v2,
        v1.stress_magnitude[size(v1.stress_magnitude)-1] as stress1,
        v2.stress_magnitude[size(v2.stress_magnitude)-1] as stress2
   WHERE stress1 > 2500 AND stress2 > 2500
   RETURN v1.grid_i, v1.grid_j, v1.grid_k, stress1,
          v2.grid_i, v2.grid_j, v2.grid_k, stress2
   LIMIT 10

24. Get complete history for critical voxel:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   WITH v, v.stress_magnitude[size(v.stress_magnitude)-1] as latest_stress
   ORDER BY latest_stress DESC
   LIMIT 1
   RETURN v.grid_i, v.grid_j, v.grid_k, v.x, v.y, v.z,
          v.stress_magnitude as stress_history,
          v.fem_timestamps as timestamps,
          size(v.stress_magnitude) as version_count,
          latest_stress

25. Monitoring coverage analysis:
   MATCH (v:Voxel)
   WITH count(v) as total_voxels
   MATCH (v2:Voxel)
   WHERE v2.stress_magnitude IS NOT NULL AND size(v2.stress_magnitude) > 0
   WITH total_voxels, count(v2) as monitored_voxels
   RETURN total_voxels, monitored_voxels,
          total_voxels - monitored_voxels as unmonitored_voxels,
          toFloat(monitored_voxels) / total_voxels * 100 as coverage_percentage

26. FEM analysis temporal resolution:
   MATCH (a:FEMAnalysis)
   RETURN a.analysis_id, a.timestamp, a.version, a.total_voxels
   ORDER BY a.timestamp ASC

27. Regional stress analysis (identify affected regions):
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   WITH v, v.stress_magnitude[size(v.stress_magnitude)-1] as stress
   WITH 
     round(v.x) as region_x,
     round(v.y) as region_y,
     round(v.z) as region_z,
     avg(stress) as avg_regional_stress,
     max(stress) as max_regional_stress,
     count(v) as voxel_count
   WHERE max_regional_stress > 2000
   RETURN region_x, region_y, region_z,
          avg_regional_stress, max_regional_stress, voxel_count
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
- Sensor properties may be null - use "IS NOT NULL" to filter for voxels with sensor data

🚨 CRITICAL FEM ARRAY ACCESS RULES:
- ALL FEM properties (stress_magnitude, eps_xx, sigma_xx, etc.) are ARRAYS
- ALWAYS check: WHERE property IS NOT NULL AND size(property) > 0
- Access latest value: property[size(property)-1]  ← MANDATORY!
- NEVER use property[-1] (not supported in Cypher)
- Stress values are in Pascals (Pa) - divide by 1000 for kPa
- Use grid_i, grid_j, grid_k for voxel identification (not just id)
- Example: v.stress_magnitude[size(v.stress_magnitude)-1] as current_stress
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
   WITH v.stress_magnitude[size(v.stress_magnitude)-1] as stress
   RETURN count(v)  ← v is not defined!
   
   RIGHT:
   MATCH (v:Voxel)
   WITH v, v.stress_magnitude[size(v.stress_magnitude)-1] as stress
   RETURN count(v), avg(stress)

3. ✅ For comprehensive statistics, use single WITH chain:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   WITH v, v.stress_magnitude[size(v.stress_magnitude)-1] as stress
   RETURN count(v) as total, avg(stress) as avg_stress, max(stress) as max_stress

4. ✅ For safety assessment, query ALL stats in ONE query:
   MATCH (v:Voxel)
   WHERE v.stress_magnitude IS NOT NULL AND size(v.stress_magnitude) > 0
   WITH v, v.stress_magnitude[size(v.stress_magnitude)-1] as stress
   RETURN 
     count(v) as total_voxels,
     avg(stress) as avg_stress,
     min(stress) as min_stress,
     max(stress) as max_stress,
     sum(CASE WHEN stress > 3000 THEN 1 ELSE 0 END) as critical_count

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
                    return json.dumps({
                        "success": True,
                        "query": cypher_query,
                        "reasoning": reasoning,
                        "results": [],
                        "message": "Query executed successfully but returned no results",
                        "attempts": attempt + 1
                    }, indent=2)
                
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
                error_msg = str(query_error)
                
                # Check if it's a syntax error that we can fix
                if "SyntaxError" in error_msg or "Variable" in error_msg or "not defined" in error_msg:
                    if attempt < max_retries:
                        print(f"🔄 Attempt {attempt + 1}/{max_retries + 1} failed, retrying with fix...")
                        # Try to fix the query
                        cypher_query = _validate_and_fix_query(
                            cypher_query, error_msg, reasoning, 
                            natural_language_query, client, openai_config
                        )
                        print(f"🔍 Retry with fixed query:\n   {cypher_query}")
                        continue
                    else:
                        print(f"❌ Max retries ({max_retries}) reached")
                        break
                else:
                    # Not a fixable error, break immediately
                    break
        
        # If we get here, all retries failed
        return json.dumps({
            "success": False,
            "error": last_error,
            "query": cypher_query,
            "reasoning": reasoning,
            "message": f"Query execution failed after {max_retries + 1} attempts",
            "attempts": max_retries + 1
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": "Failed to generate or execute query"
        }, indent=2)


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
        sample = _execute_query("MATCH (v:Voxel) RETURN v.id, v.x, v.y, v.z, v.type, v.connection_count, v.ground_connected, v.temp_c, v.strain_uE LIMIT 1")
        
        # Get voxel types
        types = _execute_query("MATCH (v:Voxel) RETURN DISTINCT v.type as type")
        
        # Count voxels with sensor data
        sensor_count = _execute_query("MATCH (v:Voxel) WHERE v.temp_c IS NOT NULL OR v.strain_uE IS NOT NULL RETURN count(v) as count")
        
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
                           v.temp_c as temp_c, v.strain_uE as strain_uE, v.load_N as load_N
                    ORDER BY v.last_updated DESC
                """
                recently_updated = _execute_query(query, {"cutoff": cutoff, "voxel_ids": voxel_ids})
            else:
                query = """
                    MATCH (v:Voxel)
                    WHERE v.last_updated > $cutoff
                    RETURN v.id as id, v.type as type,
                           v.x as x, v.y as y, v.z as z, v.last_updated as last_updated,
                           v.temp_c as temp_c, v.strain_uE as strain_uE, v.load_N as load_N
                    ORDER BY v.last_updated DESC
                    LIMIT 20
                """
                recently_updated = _execute_query(query, {"cutoff": cutoff})
        except Exception:
            pass  # last_updated may not exist on voxels
        
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
        
        result = {
            "status": "checked",
            "cutoff_time": cutoff,
            "minutes_checked": minutes,
            "checked_voxels": voxel_ids if voxel_ids else "all",
            "has_changes": len(change_notifications) > 0 or len(recently_updated) > 0,
            "total_changes": len(change_notifications) + len(recently_updated),
            "change_details": change_summary,
            "raw_change_notifications": change_notifications,
            "raw_recently_updated_voxels": recently_updated
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


# Create Agno Toolkit for Neo4j (intelligent query generation + change awareness + version history)
neo4j_toolkit = Toolkit(
    name="neo4j_tools",
    tools=[
        check_recent_updates,     # CRITICAL: Check for data changes first!
        get_property_history,     # Get version history of properties
        intelligent_query_neo4j,  # Main intelligent tool
        get_database_schema       # Helper to understand data
    ]
)


# Export for easy import
__all__ = ["neo4j_toolkit", "intelligent_query_neo4j", "get_database_schema", "check_recent_updates", "get_property_history"]

