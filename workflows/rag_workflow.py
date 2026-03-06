#!/usr/bin/env python3
"""
RAG Workflow for Agno
Orchestrates the GraphRAG query processing workflow

Workflow Steps:
1. Load memory context
2. Route question (vector search or graph query)
3. Execute appropriate search path
4. Generate context-aware response
5. Save to memory
"""

from typing import Dict, Any

# Support both package-relative and absolute imports
try:
    from ..agents.graph_rag_agent import GraphRAGAgent  # when imported as package
except ImportError:
    from agents.graph_rag_agent import GraphRAGAgent  # when running as script from repo root


class RAGWorkflow:
    """
    Main RAG workflow for question answering using Agno Framework
    
    Orchestrates the complete flow from question to answer:
    - Memory context loading (handled by Agno Agent)
    - Intelligent routing (handled by Agno Agent)
    - Hybrid retrieval (handled by Agno Agent)
    - Answer generation (handled by Agno Agent)
    - Memory persistence
    
    Lightweight with Agno framework - 75% less code!
    """
    
    def __init__(self):
        """Initialize RAG workflow with Agno agent"""
        self.rag_agent = GraphRAGAgent()
    
    def run(self, question: str, save_to_memory: bool = True) -> Dict[str, Any]:
        """
        Execute RAG workflow for a question
        
        Args:
            question: User's question
            save_to_memory: Whether to save conversation to memory
            
        Returns:
            Dictionary with answer and metadata
        """
        print(f"\n{'='*60}")
        print(f"🚀 RAG Workflow Starting")
        print(f"{'='*60}")
        
        # Execute through agent
        answer = self.rag_agent.ask(question, save_to_memory=save_to_memory)
        
        print(f"\n{'='*60}")
        print(f"✅ RAG Workflow Complete")
        print(f"{'='*60}\n")
        
        return {
            "question": question,
            "answer": answer,
            "status": "success"
        }
    
    def run_conversation(self):
        """
        Run interactive conversation loop
        
        Provides continuous Q&A interface for users
        """
        print("\n" + "="*60)
        print("🤖  GraphRAG Conversation System")
        print("="*60)
        print("Ask questions about voxels in the knowledge base!")
        print("Type 'quit', 'exit', or 'q' to exit.\n")
        
        try:
            while True:
                question = input("You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Process question
                result = self.run(question)
                print(f"\nAssistant: {result['answer']}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        finally:
            self.close()
    
    def close(self):
        """Close workflow and agent connections"""
        self.rag_agent.close()

