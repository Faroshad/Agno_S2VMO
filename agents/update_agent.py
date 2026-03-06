#!/usr/bin/env python3
"""
Update Agent for Agno
Agent for monitoring and processing incremental updates

Implements:
- JSON file monitoring
- Change detection
- Incremental graph updates
- Change notifications
"""

from typing import Dict, Any
from pathlib import Path
# Support both package-relative and absolute imports
try:
    from ..core.change_detector import (
        VoxelChangeDetector,
        IncrementalGraphUpdater,
        ChangeNotificationSystem
    )
    from ..config.settings import Settings
except ImportError:
    from core.change_detector import (
        VoxelChangeDetector,
        IncrementalGraphUpdater,
        ChangeNotificationSystem
    )
    from config.settings import Settings
from datetime import datetime


class UpdateAgent:
    """
    Agent for handling incremental updates to the voxel knowledge base
    
    Features:
    - Continuous file monitoring
    - Smart change detection
    - Targeted graph updates
    - Change notification system
    """
    
    def __init__(self, json_path: str):
        """
        Initialize Update Agent
        
        Args:
            json_path: Path to voxel JSON file to monitor
        """
        self.json_path = Path(json_path)
        
        # Initialize change detection components
        self.change_detector = VoxelChangeDetector(str(self.json_path))
        
        # Initialize graph updater
        conn_info = Settings.get_connection_info()
        self.graph_updater = IncrementalGraphUpdater(
            uri=conn_info["uri"],
            user=conn_info["user"],
            password=conn_info["password"],
            database=conn_info["database"]
        )
        
        # Initialize notification system
        self.notifier = ChangeNotificationSystem(
            uri=conn_info["uri"],
            user=conn_info["user"],
            password=conn_info["password"],
            database=conn_info["database"]
        )
        
        self.running = False
    
    def check_for_updates(self) -> Dict[str, Any]:
        """
        Check for updates in the JSON file
        
        Returns:
            Dictionary with update information
        """
        return self.change_detector.detect_changes()
    
    def process_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process detected changes and update graph
        
        Args:
            changes: Change information from detector
            
        Returns:
            Update results
        """
        if changes.get("status") != "changes_detected":
            return {"status": "no_updates"}
        
        print(f"\n🔄 Processing changes:")
        print(f"   Added: {len(changes.get('added_voxels', []))}")
        print(f"   Modified: {len(changes.get('modified_voxels', []))}")
        print(f"   Removed: {len(changes.get('removed_voxels', []))}")
        
        # Extract voxel data for changed voxels
        changed_voxel_data = {}
        new_data = changes.get("new_data", {})
        
        for voxel_id in changes.get("modified_voxels", []) + changes.get("added_voxels", []):
            if voxel_id in new_data:
                changed_voxel_data[voxel_id] = new_data[voxel_id]
        
        # Update graph database
        print("   💾 Updating database...")
        update_results = self.graph_updater.update_voxels(changes, changed_voxel_data)
        print("   ✅ Database update complete!")
        
        # Create change notifications
        timestamp = datetime.now().isoformat()
        
        # Notify about modifications (check sensor changes)
        for voxel_id in changes.get("modified_voxels", []):
            old_data = changes.get("old_data", {}).get(voxel_id, {})
            new_data_item = changes.get("new_data", {}).get(voxel_id, {})
            
            # Check for sensor property changes
            sensor_props = ['temp_c', 'strain_uE', 'load_N', 'hx711_raw']
            for prop in sensor_props:
                old_val = old_data.get(prop)
                new_val = new_data_item.get(prop)
                
                if old_val != new_val and (old_val is not None or new_val is not None):
                    self.notifier.create_change_notification(
                        voxel_id, f"{prop}_update", timestamp,
                        str(old_val), str(new_val)
                    )
        
        # Notify about additions
        for voxel_id in changes.get("added_voxels", []):
            new_data_item = changes.get("new_data", {}).get(voxel_id, {})
            voxel_type = new_data_item.get("type", "unknown")
            self.notifier.create_change_notification(
                voxel_id, "added", timestamp,
                None, f"type: {voxel_type}"
            )
        
        # Notify about removals
        for voxel_id in changes.get("removed_voxels", []):
            old_data = changes.get("old_data", {}).get(voxel_id, {})
            voxel_type = old_data.get("type", "unknown")
            self.notifier.create_change_notification(
                voxel_id, "removed", timestamp,
                f"type: {voxel_type}", None
            )
        
        print(f"\n✅ Updates complete:")
        for notification in update_results.get("notifications", []):
            print(f"   {notification}")
        
        return update_results
    
    def get_change_history(self, voxel_id: int) -> Dict[str, Any]:
        """
        Get change history for a specific voxel
        
        Args:
            voxel_id: Voxel ID
            
        Returns:
            Change history information
        """
        return self.notifier.get_voxel_change_history(voxel_id)
    
    def close(self):
        """Close all connections"""
        self.graph_updater.close()
        self.notifier.close()

