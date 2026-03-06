#!/usr/bin/env python3
"""
Vector Search Tools for Agno GraphRAG
Provides semantic search capabilities using embeddings
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from openai import OpenAI
import os
# Support both package-relative and absolute imports
try:
    from ..config.settings import Settings
except ImportError:
    from config.settings import Settings


class VectorSearchTool:
    """
    Tool for performing vector similarity search
    
    Provides agents with ability to:
    - Search for semantically similar voxels
    - Find relevant content by embedding similarity
    - Extract voxel IDs from search results
    """
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, 
                 password: Optional[str] = None, database: Optional[str] = None):
        """
        Initialize vector search tool
        
        Args:
            uri: Neo4j URI (defaults to settings)
            user: Username (defaults to settings)
            password: Password (defaults to settings)
            database: Database name (defaults to settings)
        """
        conn_info = Settings.get_connection_info()
        self.driver = GraphDatabase.driver(
            uri or conn_info["uri"],
            auth=(user or conn_info["user"], password or conn_info["password"])
        )
        self.database = database or conn_info["database"]
        
        # Initialize OpenAI for embeddings
        openai_config = Settings.get_openai_config()
        self.openai_client = OpenAI(api_key=openai_config["api_key"])
        self.embedding_model = openai_config["embedding_model"]
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of search results with voxel information
        """
        # Generate query embedding
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
        
        # Perform vector search in Neo4j
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run("""
                    CALL db.index.vector.queryNodes('voxelgraph_chunk_embeddings_index', $k, $query_embedding)
                    YIELD node, score
                    RETURN node.id as chunk_id,
                           node.text as text,
                           node.voxel_id as voxel_id,
                           score
                    ORDER BY score DESC
                """, {
                    "k": k,
                    "query_embedding": query_embedding
                })
                
                results = []
                for record in result:
                    results.append({
                        "chunk_id": record["chunk_id"],
                        "text": record["text"],
                        "voxel_id": record["voxel_id"],
                        "similarity_score": record["score"]
                    })
                
                return results
            except Exception as e:
                print(f"Vector search error: {e}")
                return []
    
    def search_with_voxel_details(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform vector search and include full voxel details
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of search results with full voxel information
        """
        # Generate query embedding
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
        
        # Perform vector search with voxel details
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run("""
                    CALL db.index.vector.queryNodes('voxelgraph_chunk_embeddings_index', $k, $query_embedding)
                    YIELD node, score
                    MATCH (v:Voxel {id: node.voxel_id})
                    RETURN node.text as chunk_text,
                           node.voxel_id as voxel_id,
                           v.type as type,
                           v.temp_c as temp_c,
                           v.strain_uE as strain_uE,
                           v.load_N as load_N,
                           v.x as x, v.y as y, v.z as z,
                           v.connection_count as connection_count,
                           score
                    ORDER BY score DESC
                """, {
                    "k": k,
                    "query_embedding": query_embedding
                })
                
                results = []
                for record in result:
                    results.append({
                        "voxel_id": record["voxel_id"],
                        "type": record["type"],
                        "temp_c": record["temp_c"],
                        "strain_uE": record["strain_uE"],
                        "load_N": record["load_N"],
                        "position": [record["x"], record["y"], record["z"]],
                        "connection_count": record["connection_count"],
                        "chunk_text": record["chunk_text"],
                        "similarity_score": record["score"]
                    })
                
                return results
            except Exception as e:
                print(f"Vector search error: {e}")
                return []
    
    def extract_voxel_ids(self, search_results: List[Dict[str, Any]]) -> List[int]:
        """
        Extract voxel IDs from search results
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            List of voxel IDs
        """
        voxel_ids = []
        for result in search_results:
            if "voxel_id" in result and result["voxel_id"] is not None:
                voxel_ids.append(result["voxel_id"])
        return voxel_ids
    
    def close(self):
        """Close database connection"""
        self.driver.close()

