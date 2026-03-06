#!/usr/bin/env python3
"""
Memory Tools for Agno GraphRAG
Provides memory access and retrieval capabilities
"""

from typing import List, Dict, Any, Optional
# Support both package-relative and absolute imports
try:
    from ..core.memory_manager import MemoryManager
    from ..config.settings import Settings
except ImportError:
    from core.memory_manager import MemoryManager
    from config.settings import Settings


class MemoryRetrievalTool:
    """
    Tool for accessing conversation and semantic memory
    
    Provides agents with ability to:
    - Retrieve conversation history
    - Search semantic memory
    - Get context for queries
    - Access memory statistics
    """
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, 
                 password: Optional[str] = None, database: Optional[str] = None):
        """
        Initialize memory retrieval tool
        
        Args:
            uri: Neo4j URI (defaults to settings)
            user: Username (defaults to settings)
            password: Password (defaults to settings)
            database: Database name (defaults to settings)
        """
        conn_info = Settings.get_connection_info()
        self.memory_manager = MemoryManager(
            uri=uri or conn_info["uri"],
            user=user or conn_info["user"],
            password=password or conn_info["password"],
            database=database or conn_info["database"]
        )
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get recent conversation history
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
        """
        return self.memory_manager.get_conversation_history(limit=limit)
    
    def search_semantic_memory(self, query: str, k: int = 5, 
                               memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search semantic memory for relevant past interactions
        
        Args:
            query: Search query
            k: Number of results to return
            memory_type: Optional filter by memory type
            
        Returns:
            List of similar memories with scores
        """
        return self.memory_manager.search_semantic_memory(query, k=k, memory_type=memory_type)
    
    def get_formatted_context(self, query: str) -> str:
        """
        Get formatted memory context for a query
        
        Args:
            query: User's query
            
        Returns:
            Formatted context string for prompts
        """
        return self.memory_manager.get_formatted_context(query)
    
    def get_context_for_query(self, query: str, include_semantic: bool = True,
                              include_conversation: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive context for a query
        
        Args:
            query: User's query
            include_semantic: Include semantic memory search
            include_conversation: Include conversation history
            
        Returns:
            Dictionary with conversation history and semantic memories
        """
        return self.memory_manager.get_context_for_query(
            query, 
            include_semantic=include_semantic,
            include_conversation=include_conversation
        )
    
    def add_conversation_turn(self, human_message: str, ai_message: str, 
                            metadata: Optional[Dict] = None):
        """
        Add a conversation turn to memory
        
        Args:
            human_message: User's message
            ai_message: AI's response
            metadata: Optional metadata
        """
        self.memory_manager.add_conversation_turn(human_message, ai_message, metadata)
    
    def add_semantic_memory(self, content: str, memory_type: str = "general", 
                           voxel_ids: Optional[List[int]] = None,
                           metadata: Optional[Dict] = None):
        """
        Add semantic memory entry
        
        Args:
            content: Memory content
            memory_type: Type of memory (qa_pair, fact, etc.)
            voxel_ids: Associated voxel IDs
            metadata: Additional metadata
        """
        self.memory_manager.add_semantic_memory(content, memory_type, voxel_ids, metadata)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories
        
        Returns:
            Dictionary with memory statistics
        """
        return self.memory_manager.get_memory_stats()
    
    def start_new_session(self) -> str:
        """
        Start a new conversation session
        
        Returns:
            New session ID
        """
        return self.memory_manager.start_new_session()
    
    def close(self):
        """Close memory manager connection"""
        self.memory_manager.close()

