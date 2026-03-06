#!/usr/bin/env python3
"""
Main Entry Point for Agno GraphRAG System

This is the primary launcher for the simulation and chatbot system:
1. Synchronized Simulation Coordinator (sensor → FEM → Neo4j)
2. GraphRAG Chatbot (natural language queries)

Usage:
    python run.py                    # Interactive menu
    python run.py --quick-test       # Quick test mode
    python run.py --sim-only         # Run simulation only
    python run.py --chat-only        # Run chatbot only
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main entry point with options"""
    parser = argparse.ArgumentParser(
        description="Agno GraphRAG System - Main Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive menu
  python run.py
  
  # Quick test (10 cycles, accelerated time)
  python run.py --quick-test
  
  # Run synchronized simulation only
  python run.py --sim-only
  
  # Run chatbot only
  python run.py --chat-only
"""
    )
    
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test mode (10 cycles, 10x speed)")
    parser.add_argument("--sim-only", action="store_true",
                       help="Run synchronized simulation only")
    parser.add_argument("--chat-only", action="store_true",
                       help="Run chatbot only")
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("🧪 Running quick test mode...")
        from synchronized_sim_coordinator import main as sim_main
        sys.argv = ["synchronized_sim_coordinator.py", "--time-scale", "0.1", "--max-cycles", "10"]
        return sim_main()
    
    elif args.sim_only:
        print("🔄 Running synchronized simulation...")
        from synchronized_sim_coordinator import main as sim_main
        return sim_main()
    
    elif args.chat_only:
        print("💬 Starting chatbot...")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
        # Clear extra args before passing to main.py
        original_argv = sys.argv
        sys.argv = ["main.py"]
        from main import main as chat_main
        try:
            return chat_main()
        finally:
            sys.argv = original_argv
    
    else:
        # Interactive menu
        print("\n" + "="*70)
        print("  AGNO GRAPHRAG SYSTEM - MAIN LAUNCHER")
        print("="*70)
        print("\nSelect mode:")
        print("  1) Run synchronized simulation only")
        print("  2) Run chatbot only")
        print("  3) Quick test (10 cycles, 10x speed)")
        print("  4) Initialize database")
        print("  0) Exit")
        print()
        
        try:
            choice = input("Enter choice [0-4]: ").strip()
            
            if choice == "1":
                from synchronized_sim_coordinator import main as sim_main
                return sim_main()
            
            elif choice == "2":
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
                original_argv = sys.argv
                sys.argv = ["main.py"]
                from main import main as chat_main
                try:
                    return chat_main()
                finally:
                    sys.argv = original_argv
            
            elif choice == "3":
                from synchronized_sim_coordinator import main as sim_main
                sys.argv = ["synchronized_sim_coordinator.py", "--time-scale", "0.1", "--max-cycles", "10"]
                return sim_main()
            
            elif choice == "4":
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
                from initialize_neo4j import main as init_main
                return init_main()
            
            elif choice == "0":
                print("Exiting...")
                return 0
            
            else:
                print("Invalid choice")
                return 1
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return 0


if __name__ == "__main__":
    sys.exit(main())

