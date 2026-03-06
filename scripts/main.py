#!/usr/bin/env python3
"""
Agno GraphRAG Main Application
Entry point for all GraphRAG operations

Commands:
- build: Build graph database from JSON
- chat: Start interactive Q&A system
- update: Run incremental update monitoring
- query: Execute single query
- history: View change history for voxel
"""

import argparse
import sys
import threading
import subprocess
import os
import platform
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__" and __package__ is None:
    project_root = str(Path(__file__).parent.parent)
    sys.path.insert(0, project_root)
    # Import as regular modules (not as package)
    from workflows.rag_workflow import RAGWorkflow
    from workflows.update_workflow import UpdateWorkflow
    from agents.update_agent import UpdateAgent
    from config.settings import Settings
else:
    from workflows.rag_workflow import RAGWorkflow
    from workflows.update_workflow import UpdateWorkflow
    from agents.update_agent import UpdateAgent
    from config.settings import Settings


def build_graph(args):
    """Build graph database from voxel grid (use initialize_neo4j.py instead)"""
    print("\n" + "="*60)
    print("🏗️  Building Graph Database")
    print("="*60)
    print("\n⚠️  Use 'python initialize_neo4j.py' to build the graph database")
    print("    This command builds from voxel_grid.npz, not JSON")
    return 1


def open_new_terminal_window(command: str, title: str = "Update Monitor"):
    """
    Open a new terminal window and run a command
    
    Args:
        command: Command to run in new terminal
        title: Window title
    """
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # Use AppleScript to open new Terminal window
        applescript = f'''
        tell application "Terminal"
            do script "{command}"
            set custom title of front window to "{title}"
            activate
        end tell
        '''
        subprocess.Popen(['osascript', '-e', applescript])
        
    elif system == "Linux":
        # Try common Linux terminal emulators
        terminals = [
            ['gnome-terminal', '--', 'bash', '-c', f'{command}; exec bash'],
            ['xterm', '-e', f'{command}; bash'],
            ['konsole', '-e', f'{command}; bash'],
            ['terminator', '-e', f'{command}; bash'],
        ]
        
        for terminal_cmd in terminals:
            try:
                subprocess.Popen(terminal_cmd)
                break
            except FileNotFoundError:
                continue
                
    elif system == "Windows":
        # Windows command
        subprocess.Popen(['start', 'cmd', '/k', command], shell=True)
    
    else:
        print(f"⚠️  Warning: Automatic terminal opening not supported on {system}")
        print(f"   Please run manually: {command}")


def start_chat(args):
    """Start interactive Q&A system with optional auto-update monitoring"""
    print("\n" + "="*60)
    print("💬 Starting Interactive Chat System")
    print("="*60)
    
    try:
        # Check if auto-update is enabled
        if hasattr(args, 'auto_update') and args.auto_update:
            json_file = args.json_file if hasattr(args, 'json_file') and args.json_file else "data/voxels/dome_voxels.json"
            
            if Path(json_file).exists():
                # Get the absolute path to main.py and json file
                main_py_path = Path(__file__).absolute()
                json_file_abs = Path(json_file).absolute()
                interval = getattr(args, 'interval', 5)
                
                # Build command to run update monitoring in new terminal
                update_command = f"cd {main_py_path.parent} && python {main_py_path} update {json_file_abs} --interval {interval}"
                
                print(f"\n🔄 Auto-update monitoring enabled")
                print(f"   Monitoring: {json_file}")
                print(f"   Check interval: {interval} seconds")
                print(f"   Mode: Separate terminal window")
                print(f"   Opening new terminal window...")
                print("="*60)
                
                # Open new terminal window with update monitoring
                open_new_terminal_window(
                    command=update_command,
                    title=f"Update Monitor - {Path(json_file).name}"
                )
                
                print("✅ Update monitor started in new terminal window\n")
                
            else:
                print(f"\n⚠️  Warning: JSON file '{json_file}' not found")
                print("   Auto-update disabled")
                print("="*60)
        
        # Create and run RAG workflow
        workflow = RAGWorkflow()
        workflow.run_conversation()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


def run_update_monitoring(args):
    """Run incremental update monitoring"""
    print("\n" + "="*60)
    print("🔄 Starting Update Monitoring")
    print("="*60)
    
    # Check if JSON file exists
    if not Path(args.json_file).exists():
        print(f"❌ Error: JSON file '{args.json_file}' not found")
        return 1
    
    try:
        # Create and run update workflow (silent mode for cleaner output)
        workflow = UpdateWorkflow(
            json_path=args.json_file,
            check_interval=args.interval,
            silent=True  # Only show updates when changes are detected
        )
        workflow.run_continuous()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


def execute_query(args):
    """Execute a single query"""
    print("\n" + "="*60)
    print("🔍 Executing Query")
    print("="*60)
    
    try:
        # Create RAG workflow
        workflow = RAGWorkflow()
        
        # Execute query
        result = workflow.run(args.question, save_to_memory=not args.no_memory)
        
        # Print answer
        print(f"\n📝 Answer:")
        print(f"{result['answer']}")
        
        # Close workflow
        workflow.close()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


def view_history(args):
    """View change history for a voxel"""
    print("\n" + "="*60)
    print(f"📜 Change History for Voxel {args.voxel_id}")
    print("="*60)
    
    try:
        # Check if JSON file exists (needed for agent initialization)
        json_file = args.json_file or "dome_voxels.json"
        if not Path(json_file).exists():
            print(f"❌ Error: JSON file '{json_file}' not found")
            return 1
        
        # Create update agent
        agent = UpdateAgent(json_file)
        
        # Get change history
        history = agent.get_change_history(args.voxel_id)
        
        if not history:
            print(f"\n✓ No change history found for voxel {args.voxel_id}")
        else:
            print(f"\n📊 Found {len(history)} changes:\n")
            for i, change in enumerate(history, 1):
                print(f"{i}. Type: {change['change_type']}")
                print(f"   Timestamp: {change['timestamp']}")
                if change.get('old_value'):
                    print(f"   Old: {change['old_value']}")
                if change.get('new_value'):
                    print(f"   New: {change['new_value']}")
                print()
        
        # Close agent
        agent.close()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


def main():
    """Main entry point - defaults to chatbot if no arguments"""
    # If no arguments provided, start chatbot directly
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("💬 Starting GraphRAG Chatbot")
        print("="*60)
        print("\n💡 Tip: Use 'python launcher.py' to start all components together\n")
        try:
            workflow = RAGWorkflow()
            workflow.run_conversation()
            return 0
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Parse arguments for advanced usage
    parser = argparse.ArgumentParser(
        description="Agno GraphRAG System - Voxel Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start chatbot (default - no arguments needed)
  python main.py
  
  # Build graph database
  python main.py build data/voxels/dome_voxels.json

  # Start interactive chat
  python main.py chat

  # Start chat with automatic update monitoring (recommended!)
  # This opens a NEW TERMINAL WINDOW for monitoring
  python main.py chat --auto-update --json-file data/voxels/dome_voxels.json

  # Start chat with auto-update and custom interval (10 seconds)
  python main.py chat --auto-update --json-file data/voxels/dome_voxels.json --interval 10

  # Run standalone update monitoring
  python main.py update data/voxels/dome_voxels.json

  # Execute single query
  python main.py query "How many voxels are there?"

  # View change history
  python main.py history 5
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build graph database from JSON")
    build_parser.add_argument("json_file", help="Path to voxel JSON file")
    build_parser.add_argument("--uri", help="Neo4j URI")
    build_parser.add_argument("--user", help="Neo4j username")
    build_parser.add_argument("--password", help="Neo4j password")
    build_parser.add_argument("--database", help="Neo4j database name")
    build_parser.add_argument("--no-clear", action="store_true", help="Don't clear database before building")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive Q&A system")
    chat_parser.add_argument("--auto-update", action="store_true", help="Enable automatic update monitoring in new terminal window")
    chat_parser.add_argument("--json-file", help="Path to JSON file for auto-update (default: data/voxels/dome_voxels.json)")
    chat_parser.add_argument("--interval", type=int, default=5, help="Check interval for auto-update in seconds (default: 5)")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Run incremental update monitoring")
    update_parser.add_argument("json_file", help="Path to voxel JSON file")
    update_parser.add_argument("--interval", type=int, default=5, help="Check interval in seconds (default: 5)")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Execute a single query")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--no-memory", action="store_true", help="Don't save to memory")
    
    # History command
    history_parser = subparsers.add_parser("history", help="View change history for voxel")
    history_parser.add_argument("voxel_id", type=int, help="Voxel ID")
    history_parser.add_argument("--json-file", help="Path to voxel JSON file (default: data/voxels/dome_voxels.json)")
    
    args = parser.parse_args()
    
    if not args.command:
        # No command specified, start chatbot
        print("\n" + "="*60)
        print("💬 Starting GraphRAG Chatbot")
        print("="*60)
        print()
        try:
            workflow = RAGWorkflow()
            workflow.run_conversation()
            return 0
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Execute command
    try:
        if args.command == "build":
            return build_graph(args)
        elif args.command == "chat":
            return start_chat(args)
        elif args.command == "update":
            return run_update_monitoring(args)
        elif args.command == "query":
            return execute_query(args)
        elif args.command == "history":
            return view_history(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

