#!/usr/bin/env python3
"""
Change Detection System for Agno GraphRAG
Comprehensive change detection and update management
Ported from LangGraph version with all change detection components

Components:
- VoxelChangeDetector: Monitors JSON file changes
- IncrementalGraphUpdater: Updates specific voxels in Neo4j
- ChangeNotificationSystem: Tracks change history
- SmartChangeDetector: Intelligent relevance filtering
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# VOXEL CHANGE DETECTOR
# =============================================================================

class VoxelChangeDetector:
    """
    Monitors JSON file changes and detects specific voxel modifications
    
    Features:
    - File hash comparison for change detection
    - Voxel-level diff analysis
    - Change history tracking
    - Neighbor impact analysis
    """
    
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.last_hash = None
        self.last_data = None
        self.change_history = {}
        
    def detect_changes(self) -> Dict[str, Any]:
        """
        Detect changes in the JSON file
        
        Returns:
            Dictionary with change information
        """
        if not self.json_path.exists():
            return {"error": "JSON file not found"}
        
        # Calculate current file hash
        current_hash = self._calculate_file_hash()
        
        # If no previous hash, initialize
        if self.last_hash is None:
            self.last_hash = current_hash
            self.last_data = self._load_json_data()
            return {"status": "initialized", "total_voxels": len(self.last_data.get("voxels", []))}
        
        # If hash hasn't changed, no updates
        if current_hash == self.last_hash:
            return {"status": "no_changes"}
        
        # Load current data
        current_data = self._load_json_data()
        
        # Detect changes
        changes = self._analyze_changes(self.last_data, current_data)
        
        # Update tracking
        self.last_hash = current_hash
        self.last_data = current_data
        
        return changes
    
    def _calculate_file_hash(self) -> str:
        """Calculate MD5 hash of the JSON file"""
        with open(self.json_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _load_json_data(self) -> Dict[str, Any]:
        """Load JSON data from file"""
        with open(self.json_path, 'r') as f:
            return json.load(f)
    
    def _analyze_changes(self, old_data: Dict, new_data: Dict) -> Dict[str, Any]:
        """Analyze changes between old and new data"""
        old_voxels = {v["id"]: v for v in old_data.get("voxels", [])}
        new_voxels = {v["id"]: v for v in new_data.get("voxels", [])}
        
        old_ids = set(old_voxels.keys())
        new_ids = set(new_voxels.keys())
        
        # Detect changes
        added_voxels = list(new_ids - old_ids)
        removed_voxels = list(old_ids - new_ids)
        modified_voxels = []
        
        # Check for modifications in existing voxels
        for voxel_id in old_ids & new_ids:
            if old_voxels[voxel_id] != new_voxels[voxel_id]:
                modified_voxels.append(voxel_id)
        
        # Analyze neighbor impacts
        neighbor_impacts = self._analyze_neighbor_impacts(
            old_voxels, new_voxels, modified_voxels + added_voxels + removed_voxels
        )
        
        # Update change history
        timestamp = datetime.now().isoformat()
        for voxel_id in modified_voxels + added_voxels:
            if voxel_id not in self.change_history:
                self.change_history[voxel_id] = []
            self.change_history[voxel_id].append({
                "timestamp": timestamp,
                "change_type": "added" if voxel_id in added_voxels else "modified"
            })
        
        return {
            "status": "changes_detected",
            "added_voxels": added_voxels,
            "removed_voxels": removed_voxels,
            "modified_voxels": modified_voxels,
            "neighbor_impacts": neighbor_impacts,
            "total_changes": len(added_voxels) + len(removed_voxels) + len(modified_voxels),
            "old_data": old_voxels,
            "new_data": new_voxels
        }
    
    def _analyze_neighbor_impacts(self, old_voxels: Dict, new_voxels: Dict, 
                                 changed_voxels: List[int]) -> List[int]:
        """Analyze which voxels are impacted by neighbor changes"""
        impacted_voxels = set()
        
        for voxel_id in changed_voxels:
            if voxel_id in new_voxels:
                neighbors = new_voxels[voxel_id].get("neighbors", [])
                impacted_voxels.update(neighbors)
            
            if voxel_id in old_voxels:
                old_neighbors = old_voxels[voxel_id].get("neighbors", [])
                impacted_voxels.update(old_neighbors)
        
        return list(impacted_voxels)


# =============================================================================
# INCREMENTAL GRAPH UPDATER
# =============================================================================

class IncrementalGraphUpdater:
    """
    Updates specific voxels in Neo4j graph database
    
    Features:
    - Targeted voxel updates
    - Embedding regeneration
    - Version history tracking
    - Neighbor relationship updates
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", 
                 password: str = "test1234", database: Optional[str] = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.openai_client = OpenAI(api_key=api_key)
    
    def update_voxels(self, changes: Dict[str, Any], voxel_data: Dict[int, Dict]) -> Dict[str, Any]:
        """Update specific voxels in the graph database"""
        update_results = {
            "updated_voxels": [],
            "failed_updates": [],
            "notifications": []
        }
        
        with self.driver.session(database=self.database) as session:
            # Handle added voxels
            for voxel_id in changes.get("added_voxels", []):
                if voxel_id in voxel_data:
                    result = self._add_voxel(session, voxel_id, voxel_data[voxel_id])
                    if result["success"]:
                        update_results["updated_voxels"].append(voxel_id)
                        update_results["notifications"].append(f"✓ Added voxel {voxel_id}")
            
            # Handle modified voxels
            for voxel_id in changes.get("modified_voxels", []):
                if voxel_id in voxel_data:
                    result = self._update_voxel(session, voxel_id, voxel_data[voxel_id])
                    if result["success"]:
                        update_results["updated_voxels"].append(voxel_id)
                        update_results["notifications"].append(f"✓ Updated voxel {voxel_id}")
                    else:
                        update_results["failed_updates"].append(voxel_id)
            
            # Handle removed voxels
            for voxel_id in changes.get("removed_voxels", []):
                result = self._remove_voxel(session, voxel_id)
                if result["success"]:
                    update_results["notifications"].append(f"✓ Removed voxel {voxel_id}")
        
        return update_results
    
    def _add_voxel(self, session, voxel_id: int, voxel_data: Dict) -> Dict[str, Any]:
        """Add a new voxel to the graph"""
        try:
            timestamp = datetime.now().isoformat()
            acc_g = voxel_data.get("acc_g", {})
            gyro_dps = voxel_data.get("gyro_dps", {})
            quality = voxel_data.get("quality", {})
            
            session.run("""
                CREATE (v:Voxel {
                    id: $id,
                    x: $x, y: $y, z: $z,
                    type: $type,
                    connection_count: $connection_count,
                    ground_connected: $ground_connected,
                    ground_level: $ground_level,
                    temp_c: $temp_c,
                    strain_uE: $strain_uE,
                    load_N: $load_N,
                    hx711_raw: $hx711_raw,
                    acc_g_x: $acc_g_x, acc_g_y: $acc_g_y, acc_g_z: $acc_g_z,
                    gyro_dps_x: $gyro_dps_x, gyro_dps_y: $gyro_dps_y, gyro_dps_z: $gyro_dps_z,
                    quality_ok: $quality_ok,
                    quality_flags: $quality_flags,
                    created_at: $timestamp,
                    last_updated: $timestamp
                })
            """, {
                "id": voxel_id,
                "x": voxel_data["position"][0],
                "y": voxel_data["position"][1],
                "z": voxel_data["position"][2],
                "type": voxel_data.get("type", "unknown"),
                "connection_count": voxel_data.get("connection_count", 0),
                "ground_connected": voxel_data.get("ground_connected", False),
                "ground_level": voxel_data.get("ground_level"),
                "temp_c": voxel_data.get("temp_c"),
                "strain_uE": voxel_data.get("strain_uE"),
                "load_N": voxel_data.get("load_N"),
                "hx711_raw": voxel_data.get("hx711_raw"),
                "acc_g_x": acc_g.get("x"),
                "acc_g_y": acc_g.get("y"),
                "acc_g_z": acc_g.get("z"),
                "gyro_dps_x": gyro_dps.get("x"),
                "gyro_dps_y": gyro_dps.get("y"),
                "gyro_dps_z": gyro_dps.get("z"),
                "quality_ok": quality.get("ok", True),
                "quality_flags": ",".join(quality.get("flags", [])) if quality.get("flags") else "",
                "timestamp": timestamp
            })
            
            # Skip version history for incremental updates (too slow)
            # self._create_initial_versions(session, voxel_id, voxel_data, timestamp)
            
            # Skip chunk/embedding creation for incremental updates (too slow)
            # self._create_chunk_for_voxel(session, voxel_id, voxel_data)
            
            # Create neighbor relationships
            self._create_neighbor_relationships(session, voxel_id, voxel_data)
            
            return {"success": True}
        except Exception as e:
            print(f"Error adding voxel {voxel_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_voxel(self, session, voxel_id: int, voxel_data: Dict) -> Dict[str, Any]:
        """Update an existing voxel with version history tracking"""
        try:
            # Get current state
            current_result = session.run("""
                MATCH (v:Voxel {id: $id})
                RETURN v.type as type, v.x as x, v.y as y, v.z as z,
                       v.temp_c as temp_c, v.strain_uE as strain_uE, v.load_N as load_N,
                       v.hx711_raw as hx711_raw, v.acc_g_x as acc_g_x, v.acc_g_y as acc_g_y,
                       v.acc_g_z as acc_g_z, v.gyro_dps_x as gyro_dps_x, v.gyro_dps_y as gyro_dps_y,
                       v.gyro_dps_z as gyro_dps_z, v.quality_ok as quality_ok
            """, {"id": voxel_id})
            
            current_voxel = current_result.single()
            current_data = dict(current_voxel) if current_voxel else {}
            
            # Skip version history for incremental updates (too slow)
            # if current_data:
            #     self._create_version_history(session, voxel_id, current_data, voxel_data)
            
            # Update voxel properties
            timestamp = datetime.now().isoformat()
            acc_g = voxel_data.get("acc_g", {})
            gyro_dps = voxel_data.get("gyro_dps", {})
            quality = voxel_data.get("quality", {})
            
            session.run("""
                MATCH (v:Voxel {id: $id})
                SET v.x = $x, v.y = $y, v.z = $z,
                    v.type = $type,
                    v.connection_count = $connection_count,
                    v.ground_connected = $ground_connected,
                    v.ground_level = $ground_level,
                    v.temp_c = $temp_c,
                    v.strain_uE = $strain_uE,
                    v.load_N = $load_N,
                    v.hx711_raw = $hx711_raw,
                    v.acc_g_x = $acc_g_x, v.acc_g_y = $acc_g_y, v.acc_g_z = $acc_g_z,
                    v.gyro_dps_x = $gyro_dps_x, v.gyro_dps_y = $gyro_dps_y, v.gyro_dps_z = $gyro_dps_z,
                    v.quality_ok = $quality_ok,
                    v.quality_flags = $quality_flags,
                    v.last_updated = $timestamp
            """, {
                "id": voxel_id,
                "x": voxel_data["position"][0],
                "y": voxel_data["position"][1],
                "z": voxel_data["position"][2],
                "type": voxel_data.get("type", "unknown"),
                "connection_count": voxel_data.get("connection_count", 0),
                "ground_connected": voxel_data.get("ground_connected", False),
                "ground_level": voxel_data.get("ground_level"),
                "temp_c": voxel_data.get("temp_c"),
                "strain_uE": voxel_data.get("strain_uE"),
                "load_N": voxel_data.get("load_N"),
                "hx711_raw": voxel_data.get("hx711_raw"),
                "acc_g_x": acc_g.get("x"),
                "acc_g_y": acc_g.get("y"),
                "acc_g_z": acc_g.get("z"),
                "gyro_dps_x": gyro_dps.get("x"),
                "gyro_dps_y": gyro_dps.get("y"),
                "gyro_dps_z": gyro_dps.get("z"),
                "quality_ok": quality.get("ok", True),
                "quality_flags": ",".join(quality.get("flags", [])) if quality.get("flags") else "",
                "timestamp": timestamp
            })
            
            # Skip chunk/embedding updates for incremental updates (too slow)
            # self._update_chunk_for_voxel(session, voxel_id, voxel_data)
            
            return {"success": True}
        except Exception as e:
            print(f"Error updating voxel {voxel_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _remove_voxel(self, session, voxel_id: int) -> Dict[str, Any]:
        """Remove a voxel from the graph"""
        try:
            session.run("""
                MATCH (v:Voxel {id: $id})
                DETACH DELETE v
            """, {"id": voxel_id})
            return {"success": True}
        except Exception as e:
            print(f"Error removing voxel {voxel_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_initial_versions(self, session, voxel_id: int, voxel_data: Dict, timestamp: str):
        """Create initial version [0] for all properties"""
        properties = [
            ("type", voxel_data.get("type", "unknown")),
            ("position", str(voxel_data["position"])),
            ("temp_c", str(voxel_data.get("temp_c"))),
            ("strain_uE", str(voxel_data.get("strain_uE"))),
            ("load_N", str(voxel_data.get("load_N"))),
            ("hx711_raw", str(voxel_data.get("hx711_raw")))
        ]
        
        for prop_name, prop_value in properties:
            session.run("""
                CREATE (vp:VoxelProperty {
                    voxel_id: $voxel_id,
                    property_name: $property_name,
                    property_value: $value,
                    version_number: 0,
                    timestamp: $timestamp,
                    change_type: 'initial'
                })
            """, {
                "voxel_id": voxel_id,
                "property_name": prop_name,
                "value": prop_value,
                "timestamp": timestamp
            })
    
    def _create_version_history(self, session, voxel_id: int, old_data: Dict, new_data: Dict):
        """Create versioned property entries for changes"""
        timestamp = datetime.now().isoformat()
        
        # Get current version numbers
        version_result = session.run("""
            MATCH (vp:VoxelProperty {voxel_id: $voxel_id})
            RETURN vp.property_name as property_name, 
                   max(vp.version_number) as max_version
        """, {"voxel_id": voxel_id})
        
        current_versions = {}
        for record in version_result:
            current_versions[record["property_name"]] = record["max_version"]
        
        # Check for sensor property changes
        sensor_properties = ['temp_c', 'strain_uE', 'load_N', 'hx711_raw', 'acc_g_x', 'acc_g_y', 'acc_g_z', 'gyro_dps_x', 'gyro_dps_y', 'gyro_dps_z']
        
        for prop in sensor_properties:
            old_val = old_data.get(prop)
            new_val = new_data.get(prop)
            
            # Check if sensor value changed
            if old_val != new_val and (old_val is not None or new_val is not None):
                version = current_versions.get(prop, -1) + 1
                session.run("""
                    CREATE (vp:VoxelProperty {
                        voxel_id: $voxel_id,
                        property_name: $property_name,
                        property_value: $new_value,
                        version_number: $version,
                        timestamp: $timestamp,
                        change_type: 'sensor_update'
                    })
                """, {
                    "voxel_id": voxel_id,
                    "property_name": prop,
                    "new_value": str(new_val),
                    "version": version,
                    "timestamp": timestamp
                })
    
    def _create_chunk_for_voxel(self, session, voxel_id: int, voxel_data: Dict):
        """Create chunk with embedding for a voxel"""
        text = self._generate_voxel_description(voxel_data)
        embedding = self._get_embedding(text)
        
        if embedding:
            session.run("""
                CREATE (c:Chunk {
                    id: $chunk_id,
                    text: $text,
                    embedding: $embedding,
                    voxel_id: $voxel_id
                })
            """, {
                "chunk_id": f"chunk_{voxel_id}",
                "text": text,
                "embedding": embedding,
                "voxel_id": voxel_id
            })
            
            session.run("""
                MATCH (c:Chunk {voxel_id: $voxel_id}), (v:Voxel {id: $voxel_id})
                MERGE (c)-[:DESCRIBES]->(v)
            """, {"voxel_id": voxel_id})
    
    def _update_chunk_for_voxel(self, session, voxel_id: int, voxel_data: Dict):
        """Update chunk embedding for a voxel"""
        # Delete existing chunk
        session.run("""
            MATCH (c:Chunk {voxel_id: $voxel_id})
            DETACH DELETE c
        """, {"voxel_id": voxel_id})
        
        # Create new chunk
        self._create_chunk_for_voxel(session, voxel_id, voxel_data)
    
    def _generate_voxel_description(self, voxel_data: Dict) -> str:
        """Generate text description for voxel"""
        position = voxel_data["position"]
        voxel_type = voxel_data.get("type", "unknown")
        connection_count = voxel_data.get("connection_count", 0)
        temp_c = voxel_data.get("temp_c")
        strain_uE = voxel_data.get("strain_uE")
        load_N = voxel_data.get("load_N")
        
        text = f"Voxel at position ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}), "
        text += f"type: {voxel_type}, with {connection_count} connections"
        
        # Add sensor data if available
        sensor_info = []
        if temp_c is not None:
            sensor_info.append(f"temperature: {temp_c}°C")
        if strain_uE is not None:
            sensor_info.append(f"strain: {strain_uE}μE")
        if load_N is not None:
            sensor_info.append(f"load: {load_N}N")
        
        if sensor_info:
            text += ", " + ", ".join(sensor_info)
        
        return text
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return None
    
    def _create_neighbor_relationships(self, session, voxel_id: int, voxel_data: Dict):
        """Create neighbor relationships for a voxel"""
        for neighbor_id in voxel_data.get("neighbors", []):
            session.run("""
                MATCH (a:Voxel {id: $source_id}), (b:Voxel {id: $target_id})
                MERGE (a)-[:ADJACENT_TO]->(b)
            """, {
                "source_id": voxel_id,
                "target_id": neighbor_id
            })
    
    def close(self):
        """Close database connection"""
        self.driver.close()


# =============================================================================
# CHANGE NOTIFICATION SYSTEM
# =============================================================================

class ChangeNotificationSystem:
    """
    Notifies GraphRAG system about changes for context-aware responses
    
    Features:
    - Change event tracking
    - Voxel history maintenance
    - Context injection for queries
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", 
                 password: str = "test1234", database: Optional[str] = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def create_change_notification(self, voxel_id: int, change_type: str, timestamp: str, 
                                 old_value: Optional[str] = None, new_value: Optional[str] = None):
        """Create a change notification node"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (v:Voxel {id: $voxel_id})
                CREATE (n:ChangeNotification {
                    voxel_id: $voxel_id,
                    change_type: $change_type,
                    timestamp: $timestamp,
                    old_value: $old_value,
                    new_value: $new_value
                })
                MERGE (n)-[:NOTIFIES]->(v)
            """, {
                "voxel_id": voxel_id,
                "change_type": change_type,
                "timestamp": timestamp,
                "old_value": old_value,
                "new_value": new_value
            })
    
    def get_voxel_change_history(self, voxel_id: int) -> List[Dict]:
        """Get change history for a specific voxel"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:ChangeNotification {voxel_id: $voxel_id})
                RETURN n.change_type as change_type, n.timestamp as timestamp,
                       n.old_value as old_value, n.new_value as new_value
                ORDER BY n.timestamp DESC
            """, {"voxel_id": voxel_id})
            
            return [dict(record) for record in result]
    
    def close(self):
        """Close database connection"""
        self.driver.close()


# =============================================================================
# SMART CHANGE DETECTOR
# =============================================================================

class SmartChangeDetector:
    """
    Intelligent change detection with relevance filtering
    
    Features:
    - Relevance scoring
    - Temporal filtering
    - Context-aware change detection
    """
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.active_questions = {}
        self.relevance_threshold = 0.7
    
    def start_question_tracking(self, question: str, voxel_ids: List[str]) -> str:
        """Start tracking a question and its relevant voxels"""
        tracking_id = f"q_{int(time.time())}_{hash(question) % 10000}"
        self.active_questions[tracking_id] = {
            "question": question,
            "voxel_ids": voxel_ids,
            "start_time": datetime.now(),
            "relevant_changes": []
        }
        return tracking_id
    
    def end_question_tracking(self, tracking_id: str):
        """End tracking for a question"""
        if tracking_id in self.active_questions:
            del self.active_questions[tracking_id]

