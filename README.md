# Agno GraphRAG: Voxel Knowledge Graph System

A production-ready graph database system for managing voxelized structural data with real-time change awareness, complete version history tracking, and intelligent natural language querying.

## Overview

This system implements a novel never-delete architecture for temporal data management, enabling incremental updates with significant performance improvements while maintaining high query accuracy through multi-layer reasoning. Built on Neo4j graph database with Agno framework integration.

## Key Features

- **Never-Delete Version Architecture**: Complete temporal data model with property versioning
- **Incremental Updates**: Optimized performance for continuous data refresh
- **5-Step Reasoning Chain**: Multi-layer query analysis for high accuracy
- **Automatic Change Detection**: Real-time notifications with context injection
- **Intelligent Query Generation**: LLM-powered Cypher generation
- **Complete Historical Access**: Query interface for any property version
- **Structural Health Monitoring**: FEM integration with sensor-driven updates
- **GraphRAG**: Hybrid graph-vector retrieval for natural language queries

## Architecture

### Three-Tier System

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│  GraphRAG Agent │ Update Agent │ FEM Analysis           │
└─────────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Tool Layer                            │
│  Query Generation │ Change Detection │ Version History  │
└─────────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Storage Layer                          │
│              Neo4j Graph Database                        │
│  Voxels │ VoxelProperty (Versioned) │ Change Notifications │
└─────────────────────────────────────────────────────────┘
```

### Data Model

- **Voxel Nodes**: Primary structural elements with current state
- **VoxelProperty Nodes**: Versioned history of all property changes
- **ChangeNotification Nodes**: Recent change events for agent awareness
- **FEMAnalysis Nodes**: Finite Element Method simulation results
- **Chunk Nodes**: Semantic embeddings for vector search

## Installation

### Prerequisites

- Python 3.8+
- Neo4j 5.0+
- 8GB RAM minimum (16GB recommended)
- GPU optional but recommended for FEM analysis

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd agno_graphrag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export OPENAI_API_KEY="your_key"
```

4. Initialize the database:
```bash
python run.py
# Select option 5 to initialize database
```

## Usage

### Main Launcher (Recommended)

The simplest way to run the system:

```bash
python run.py
```

This provides an interactive menu with options:
1. Run synchronized simulation only
2. Run chatbot only
3. Quick test mode
4. Initialize database

### Quick Start Options

```bash
# Quick test (10 cycles, accelerated)
python run.py --quick-test

# Run simulation only
python run.py --sim-only

# Run chatbot only
python run.py --chat-only
```

Example queries:
- "What is the maximum stress in the structure?"
- "Show me voxels with stress above 500kPa"
- "What are the current sensor readings?"
- "Which voxels have the highest strain?"
- "What changed in the last 10 minutes?"

### Advanced Usage

#### Direct Component Access

```bash
# Run synchronized simulation with custom parameters
python synchronized_sim_coordinator.py --period 60 --max-cycles 100

# Run chatbot directly
python scripts/main.py

# Initialize database
python scripts/initialize_neo4j.py
```

## System Components

### Core Modules

- **`agents/`**: AI agents for GraphRAG queries and updates
- **`tools/`**: Core tools (FEM analysis, Neo4j interface, memory, vector search)
- **`core/`**: Core functionality (change detection, memory management, Neo4j converters)
- **`workflows/`**: Workflow orchestration (RAG, updates)
- **`config/`**: System configuration and settings

### Supporting Modules

- **`sensors/`**: Realistic sensor data generator with environmental scenarios
- **`models/`**: Neural network models (S2V, UNet) and mesh geometry
- **`utils/`**: Utility scripts (voxelizer, sensor positions)

## Performance

### Metrics

- **Update Performance**: Optimized incremental updates
- **Query Accuracy**: 90%+ accuracy in contextual queries
- **Change Detection**: Sub-second latency
- **Version History**: Complete temporal access

### Scalability

- Tested with 32,000+ voxels
- Handles 100+ simultaneous versions
- Memory efficient batch processing
- GPU-accelerated FEM calculations

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

Built with:
- [Agno Framework](https://agno.com)
- [Neo4j](https://neo4j.com)
- [OpenAI](https://openai.com)
