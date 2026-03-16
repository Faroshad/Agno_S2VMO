#!/usr/bin/env python3
"""
Configuration Settings for Agno GraphRAG System
Centralized configuration management
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """
    Centralized configuration for Agno GraphRAG system
    
    Configuration areas:
    - Neo4j database connection
    - OpenAI API settings
    - Memory management
    - Agent behavior
    - File paths and storage
    """
    
    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "test1234")
    NEO4J_DATABASE: Optional[str] = os.getenv("NEO4J_DATABASE", None)
    
    # Namespace for graph nodes (optional)
    GRAPH_NAMESPACE: str = "VoxelGraph"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")  # GPT-4 Turbo with 128K context
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_TEMPERATURE: float = 0.0
    
    # Memory Configuration
    MEMORY_BUFFER_SIZE: int = 10  # Number of recent messages to keep in buffer
    CONVERSATION_CONTEXT_MESSAGES: int = 20  # Max recent messages injected into prompt context
    SEMANTIC_MEMORY_TOP_K: int = 5  # Number of semantic memories to retrieve
    MEMORY_RELEVANCE_THRESHOLD: float = 0.7  # Minimum similarity score
    
    # Agent Configuration
    VECTOR_SEARCH_TOP_K: int = 3  # Number of vector search results
    MAX_RETRIES: int = 3  # Max retries for API calls
    RETRY_DELAY: float = 1.0  # Seconds between retries
    
    # Query Validation Configuration
    MAX_QUERY_RETRIES: int = 2  # Max retries for Cypher query generation
    ENABLE_QUERY_VALIDATION: bool = True  # Enable automatic query validation and retry
    
    # File Monitoring Configuration
    UPDATE_CHECK_INTERVAL: int = 5  # Seconds between file change checks

    # Runtime supervision thresholds
    MAX_CYCLE_OVERRUN_FACTOR: float = 1.2  # Cycle considered overloaded if duration > period * factor

    # Structural alert thresholds (Pascals)
    STRESS_WARNING_THRESHOLD_PA: float = float(os.getenv("STRESS_WARNING_THRESHOLD_PA", "2000"))
    STRESS_CRITICAL_THRESHOLD_PA: float = float(os.getenv("STRESS_CRITICAL_THRESHOLD_PA", "3000"))
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    
    # Voxel Type Definitions (for documentation and prompts)
    VOXEL_TYPES: dict = {
        "joint": "Structural connection points where multiple elements meet (high connectivity)",
        "beam": "Linear structural elements that span between joints (lower connectivity)"
    }
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration settings
        
        Returns:
            True if configuration is valid
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required in environment variables")
        
        if not cls.NEO4J_PASSWORD:
            raise ValueError("NEO4J_PASSWORD is required")
        
        return True
    
    @classmethod
    def get_connection_info(cls) -> dict:
        """
        Get Neo4j connection information
        
        Returns:
            Dictionary with connection parameters
        """
        return {
            "uri": cls.NEO4J_URI,
            "user": cls.NEO4J_USER,
            "password": cls.NEO4J_PASSWORD,
            "database": cls.NEO4J_DATABASE
        }
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """
        Get OpenAI configuration
        
        Returns:
            Dictionary with OpenAI settings
        """
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "embedding_model": cls.OPENAI_EMBEDDING_MODEL,
            "temperature": cls.OPENAI_TEMPERATURE
        }

    @classmethod
    def get_alert_thresholds(cls) -> dict:
        """Get structural alert threshold configuration."""
        return {
            "stress_warning_pa": cls.STRESS_WARNING_THRESHOLD_PA,
            "stress_critical_pa": cls.STRESS_CRITICAL_THRESHOLD_PA,
        }


# Validate settings on import
Settings.validate()

