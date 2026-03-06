#!/usr/bin/env python3
"""
Update Workflow for Agno
Orchestrates the incremental update monitoring workflow

Workflow Steps:
1. Monitor JSON file for changes
2. Detect specific voxel modifications
3. Update graph database incrementally
4. Create change notifications
"""

import time
from typing import Dict, Any
# Support both package-relative and absolute imports
try:
    from ..agents.update_agent import UpdateAgent
except ImportError:
    from agents.update_agent import UpdateAgent


class UpdateWorkflow:
    """
    Workflow for continuous monitoring and incremental updates
    
    Features:
    - Continuous file monitoring
    - Automatic change detection
    - Incremental graph updates
    - Change notification system
    """
    
    def __init__(self, json_path: str, check_interval: int = 5, silent: bool = False):
        """
        Initialize update workflow
        
        Args:
            json_path: Path to voxel JSON file to monitor
            check_interval: Seconds between checks
            silent: If True, only show messages when changes detected (for background mode)
        """
        self.update_agent = UpdateAgent(json_path)
        self.check_interval = check_interval
        self.silent = silent
        self.running = False
    
    def run_once(self, silent: bool = None) -> Dict[str, Any]:
        """
        Run a single update check
        
        Args:
            silent: Override instance silent mode for this check
            
        Returns:
            Update results
        """
        # Use instance silent mode if not overridden
        is_silent = silent if silent is not None else self.silent
        
        if not is_silent:
            print("\n🔍 Checking for updates...")
        
        # Check for changes
        changes = self.update_agent.check_for_updates()
        
        if changes.get("status") == "changes_detected":
            # Process changes
            results = self.update_agent.process_changes(changes)
            
            # Always show when changes detected, even in silent mode
            if is_silent:
                print(f"\n🔄 [Background Update] Changes detected!")
                added = changes.get('added_count', 0)
                removed = changes.get('removed_count', 0)
                modified = changes.get('modified_count', 0)
                
                details = []
                if added > 0:
                    details.append(f"{added} added")
                if removed > 0:
                    details.append(f"{removed} removed")
                if modified > 0:
                    details.append(f"{modified} modified")
                
                print(f"   {', '.join(details)}")
                print(f"   ✅ Graph updated at {time.strftime('%H:%M:%S')}")
            
            return {
                "status": "updated",
                "changes": changes,
                "results": results
            }
        elif changes.get("status") == "initialized":
            if not is_silent:
                print(f"✓ Initialized with {changes.get('total_voxels', 0)} voxels")
            return {"status": "initialized"}
        else:
            if not is_silent:
                print("✓ No changes detected")
            return {"status": "no_changes"}
    
    def run_continuous(self, silent: bool = None):
        """
        Run continuous monitoring loop
        
        Args:
            silent: If True, only show messages when changes detected
        
        Monitors file and processes updates automatically
        """
        self.running = True
        is_silent = silent if silent is not None else self.silent
        
        if not is_silent:
            print("\n" + "="*60)
            print("🔄 Agno GraphRAG Update Monitoring")
            print("="*60)
            print(f"Monitoring: {self.update_agent.json_path}")
            print(f"Check interval: {self.check_interval} seconds")
            print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                result = self.run_once(silent=is_silent)
                
                if not is_silent and result["status"] == "updated":
                    print(f"\n✅ Update completed at {time.strftime('%H:%M:%S')}")
                
                # Wait before next check
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            if not is_silent:
                print("\n\n🛑 Stopping monitoring...")
            self.running = False
        finally:
            self.close()
    
    def stop(self):
        """Stop continuous monitoring"""
        self.running = False
    
    def close(self):
        """Close workflow and agent connections"""
        self.update_agent.close()

