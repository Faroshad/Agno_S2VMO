"""
Agno GraphRAG - Voxel Knowledge Base System
A lightweight, high-performance GraphRAG system built with Agno framework

Components:
- agents: Intelligent agents for querying and updating
- tools: Neo4j, vector search, and memory tools
- workflows: RAG and update workflows
- core: Graph building, memory management, change detection
- config: Settings and configuration
"""

__version__ = "1.0.0"
__author__ = "GraphRAG Team"
__description__ = "Agno-based GraphRAG system for voxel knowledge bases"

from .agents.graph_rag_agent import GraphRAGAgent
from .agents.update_agent import UpdateAgent
from .workflows.rag_workflow import RAGWorkflow
from .workflows.update_workflow import UpdateWorkflow
from .core.memory_manager import MemoryManager
from .core.voxel_grid_to_neo4j import VoxelGridToNeo4j

__all__ = [
    "GraphRAGAgent",
    "UpdateAgent",
    "RAGWorkflow",
    "UpdateWorkflow",
    "MemoryManager",
    "VoxelGridToNeo4j"
]

