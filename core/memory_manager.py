#!/usr/bin/env python3
"""
Memory Management System for Agno GraphRAG
Provides conversation memory and embedding memory with Neo4j persistence
Ported from LangGraph version with Agno compatibility

Features:
- Conversation history tracking
- Semantic memory storage with embeddings
- Neo4j persistence for long-term memory
- Session-based memory management
- Memory retrieval and search capabilities
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class MemoryManager:
    """
    Comprehensive memory management system with Neo4j persistence
    
    Manages:
    - Conversation history (short-term memory)
    - Semantic memory with embeddings (long-term memory)
    - Session-based memory tracking
    - Memory persistence in Neo4j
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", 
                 password: str = "test1234", database: Optional[str] = None):
        """
        Initialize memory manager with Neo4j connection
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
        # Initialize OpenAI for embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.openai_client = OpenAI(api_key=api_key)
        
        # Track current session
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now().isoformat()
        
        # In-memory conversation buffer
        self.conversation_buffer: List[Dict[str, str]] = []
        
        # Initialize memory schema in Neo4j
        self._initialize_memory_schema()
    
    def _initialize_memory_schema(self):
        """Initialize Neo4j schema for memory storage"""
        with self.driver.session(database=self.database) as session:
            # Create constraints and indexes for efficient memory retrieval
            try:
                session.run("""
                    CREATE CONSTRAINT conversation_session_id IF NOT EXISTS
                    FOR (c:ConversationSession) REQUIRE c.session_id IS UNIQUE
                """)
                
                session.run("""
                    CREATE INDEX conversation_timestamp IF NOT EXISTS
                    FOR (m:ConversationMessage) ON (m.timestamp)
                """)
                
                session.run("""
                    CREATE INDEX semantic_memory_timestamp IF NOT EXISTS
                    FOR (sm:SemanticMemory) ON (sm.timestamp)
                """)
                
                # Create vector index for semantic memory search
                session.run("""
                    CREATE VECTOR INDEX semantic_memory_embeddings IF NOT EXISTS
                    FOR (sm:SemanticMemory) ON (sm.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1536,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
                
                print("✓ Memory schema initialized")
            except Exception as e:
                # Schema may already exist
                pass
    
    # =============================================================================
    # CONVERSATION MEMORY (SHORT-TERM)
    # =============================================================================
    
    def add_conversation_turn(self, human_message: str, ai_message: str, 
                            metadata: Optional[Dict] = None):
        """
        Add a conversation turn to memory
        
        Args:
            human_message: User's message
            ai_message: AI's response
            metadata: Optional metadata (voxel_ids, query_type, etc.)
        """
        # Add to in-memory buffer
        self.conversation_buffer.append({"role": "human", "content": human_message})
        self.conversation_buffer.append({"role": "ai", "content": ai_message})
        
        # Persist to Neo4j
        self._persist_conversation_turn(human_message, ai_message, metadata)
    
    def _persist_conversation_turn(self, human_message: str, ai_message: str, 
                                   metadata: Optional[Dict] = None):
        """Persist conversation turn to Neo4j"""
        with self.driver.session(database=self.database) as session:
            timestamp = datetime.now().isoformat()
            
            # Create or get session node
            session.run("""
                MERGE (s:ConversationSession {session_id: $session_id})
                ON CREATE SET s.created_at = $timestamp,
                              s.message_count = 0
                SET s.last_updated = $timestamp,
                    s.message_count = s.message_count + 2
            """, {
                "session_id": self.session_id,
                "timestamp": timestamp
            })
            
            # Create human message node
            session.run("""
                MATCH (s:ConversationSession {session_id: $session_id})
                CREATE (m:ConversationMessage {
                    message_id: $human_id,
                    role: 'human',
                    content: $human_message,
                    timestamp: $timestamp,
                    metadata: $metadata
                })
                CREATE (s)-[:HAS_MESSAGE]->(m)
            """, {
                "session_id": self.session_id,
                "human_id": str(uuid.uuid4()),
                "human_message": human_message,
                "timestamp": timestamp,
                "metadata": json.dumps(metadata or {})
            })
            
            # Create AI message node
            session.run("""
                MATCH (s:ConversationSession {session_id: $session_id})
                CREATE (m:ConversationMessage {
                    message_id: $ai_id,
                    role: 'ai',
                    content: $ai_message,
                    timestamp: $timestamp,
                    metadata: $metadata
                })
                CREATE (s)-[:HAS_MESSAGE]->(m)
            """, {
                "session_id": self.session_id,
                "ai_id": str(uuid.uuid4()),
                "ai_message": ai_message,
                "timestamp": timestamp,
                "metadata": json.dumps(metadata or {})
            })
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation history from memory buffer
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
        """
        if limit:
            return self.conversation_buffer[-limit:]
        return self.conversation_buffer
    
    def get_conversation_buffer(self) -> str:
        """
        Get formatted conversation buffer for context
        
        Returns:
            Formatted conversation history string
        """
        if not self.conversation_buffer:
            return ""
        
        formatted_lines = []
        for msg in self.conversation_buffer:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted_lines.append(f"{role}: {content}")
        
        return "\n".join(formatted_lines)
    
    def clear_conversation_memory(self):
        """Clear conversation memory buffer (but keeps Neo4j persistence)"""
        self.conversation_buffer = []
    
    # =============================================================================
    # SEMANTIC MEMORY (LONG-TERM WITH EMBEDDINGS)
    # =============================================================================
    
    def add_semantic_memory(self, content: str, memory_type: str = "general", 
                           voxel_ids: Optional[List[int]] = None,
                           metadata: Optional[Dict] = None):
        """
        Add semantic memory with embedding
        
        Args:
            content: Memory content (question-answer pair, fact, etc.)
            memory_type: Type of memory (qa_pair, fact, insight, etc.)
            voxel_ids: Associated voxel IDs
            metadata: Additional metadata
        """
        # Generate embedding
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=content
            )
            embedding = response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return
        
        # Store in Neo4j
        with self.driver.session(database=self.database) as session:
            timestamp = datetime.now().isoformat()
            memory_id = str(uuid.uuid4())
            
            session.run("""
                CREATE (sm:SemanticMemory {
                    memory_id: $memory_id,
                    content: $content,
                    memory_type: $memory_type,
                    embedding: $embedding,
                    voxel_ids: $voxel_ids,
                    timestamp: $timestamp,
                    session_id: $session_id,
                    metadata: $metadata
                })
            """, {
                "memory_id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "embedding": embedding,
                "voxel_ids": voxel_ids or [],
                "timestamp": timestamp,
                "session_id": self.session_id,
                "metadata": json.dumps(metadata or {})
            })
            
            # Link to voxels if provided
            if voxel_ids:
                for voxel_id in voxel_ids:
                    try:
                        session.run("""
                            MATCH (sm:SemanticMemory {memory_id: $memory_id})
                            MATCH (v:Voxel {id: $voxel_id})
                            MERGE (sm)-[:RELATES_TO]->(v)
                        """, {
                            "memory_id": memory_id,
                            "voxel_id": voxel_id
                        })
                    except Exception:
                        pass  # Voxel might not exist
    
    def search_semantic_memory(self, query: str, k: int = 5, 
                               memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search semantic memory using vector similarity
        
        Args:
            query: Search query
            k: Number of results to return
            memory_type: Filter by memory type
            
        Returns:
            List of similar memories with scores
        """
        # Generate query embedding
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []
        
        with self.driver.session(database=self.database) as session:
            # Build query with optional type filter
            type_filter = "AND sm.memory_type = $memory_type" if memory_type else ""
            
            try:
                result = session.run(f"""
                    CALL db.index.vector.queryNodes('semantic_memory_embeddings', $k, $query_embedding)
                    YIELD node, score
                    WHERE score > 0.7 {type_filter}
                    RETURN node.memory_id as memory_id,
                           node.content as content,
                           node.memory_type as memory_type,
                           node.voxel_ids as voxel_ids,
                           node.timestamp as timestamp,
                           score
                    ORDER BY score DESC
                    LIMIT $k
                """, {
                    "k": k,
                    "query_embedding": query_embedding,
                    "memory_type": memory_type
                })
                
                memories = []
                for record in result:
                    memories.append({
                        "memory_id": record["memory_id"],
                        "content": record["content"],
                        "memory_type": record["memory_type"],
                        "voxel_ids": record["voxel_ids"],
                        "timestamp": record["timestamp"],
                        "similarity_score": record["score"]
                    })
                
                return memories
            except Exception as e:
                print(f"Error searching semantic memory: {e}")
                return []
    
    # =============================================================================
    # SESSION MANAGEMENT
    # =============================================================================
    
    def start_new_session(self) -> str:
        """
        Start a new conversation session
        
        Returns:
            New session ID
        """
        # Save current session summary if it has messages
        if self.conversation_buffer:
            self._save_session_summary()
        
        # Clear in-memory conversation
        self.clear_conversation_memory()
        
        # Generate new session
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now().isoformat()
        
        return self.session_id
    
    def _save_session_summary(self):
        """Save a summary of the current session as semantic memory"""
        if not self.conversation_buffer:
            return
        
        # Create session summary
        summary_parts = []
        for i, msg in enumerate(self.conversation_buffer[-10:]):  # Last 10 messages
            role = msg["role"].capitalize()
            summary_parts.append(f"{role}: {msg['content'][:100]}")
        
        summary = "\n".join(summary_parts)
        
        # Store as semantic memory
        self.add_semantic_memory(
            content=summary,
            memory_type="session_summary",
            metadata={"session_id": self.session_id, "message_count": len(self.conversation_buffer)}
        )
    
    # =============================================================================
    # MEMORY RETRIEVAL FOR CONTEXT
    # =============================================================================
    
    def get_context_for_query(self, query: str, include_semantic: bool = True,
                              include_conversation: bool = True) -> Dict[str, Any]:
        """
        Get relevant context for a query from all memory sources
        
        Args:
            query: User's query
            include_semantic: Include semantic memory search
            include_conversation: Include conversation history
            
        Returns:
            Dictionary with relevant context
        """
        context = {}
        
        # Get conversation history
        if include_conversation:
            context["conversation_history"] = self.get_conversation_history(limit=5)
        
        # Search semantic memory
        if include_semantic:
            context["semantic_memories"] = self.search_semantic_memory(query, k=3)
        
        return context
    
    def get_formatted_context(self, query: str) -> str:
        """
        Get formatted context string for prompts
        
        Args:
            query: User's query
            
        Returns:
            Formatted context string
        """
        context = self.get_context_for_query(query)
        
        parts = []
        
        # Add conversation history
        if context.get("conversation_history"):
            parts.append("Recent Conversation:")
            for msg in context["conversation_history"]:
                role = msg["role"].capitalize()
                content = msg["content"][:100]
                parts.append(f"  {role}: {content}")
        
        # Add semantic memories
        if context.get("semantic_memories"):
            parts.append("\nRelevant Past Knowledge:")
            for memory in context["semantic_memories"]:
                content = memory['content'][:150]
                score = memory['similarity_score']
                parts.append(f"  - {content} (similarity: {score:.2f})")
        
        return "\n".join(parts) if parts else "No relevant context found"
    
    # =============================================================================
    # UTILITIES
    # =============================================================================
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        with self.driver.session(database=self.database) as session:
            # Count conversation messages
            conv_result = session.run("""
                MATCH (m:ConversationMessage)
                RETURN count(m) as total_messages
            """)
            total_messages = conv_result.single()["total_messages"]
            
            # Count semantic memories
            sem_result = session.run("""
                MATCH (sm:SemanticMemory)
                RETURN count(sm) as total_memories,
                       count(DISTINCT sm.memory_type) as memory_types
            """)
            sem_record = sem_result.single()
            
            # Count sessions
            session_result = session.run("""
                MATCH (s:ConversationSession)
                RETURN count(s) as total_sessions
            """)
            total_sessions = session_result.single()["total_sessions"]
            
            return {
                "total_conversation_messages": total_messages,
                "total_semantic_memories": sem_record["total_memories"],
                "memory_types": sem_record["memory_types"],
                "total_sessions": total_sessions,
                "current_session_id": self.session_id,
                "current_session_messages": len(self.conversation_buffer)
            }
    
    def close(self):
        """Close database connection and save session"""
        # Save session summary before closing
        if self.conversation_buffer:
            self._save_session_summary()
        
        self.driver.close()

