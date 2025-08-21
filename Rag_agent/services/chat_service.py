import os
from pathlib import Path
from simple_agent import SimpleWorkingAgent

# Cache for agents by user_id
_agent_cache = {}

def get_query_response(query: str, user_id: str, clear_history: bool = False) -> str:
    """
    Get response from the agent for a given query.
    
    Args:
        query: The user's question
        user_id: User identifier for chat history
        clear_history: Whether to clear chat history
    
    Returns:
        Agent's response as string
    """
    try:
        # Set up paths
        BASE_DIR = Path(__file__).resolve().parent.parent
        data_dir = str(BASE_DIR / "dataStorage")
        storage_dir = str(BASE_DIR / "indexStorage")
        
        # Get or create agent for this user
        if user_id not in _agent_cache:
            print(f"Creating new agent for user: {user_id}")
            _agent_cache[user_id] = SimpleWorkingAgent(data_dir, storage_dir, user_id)
        
        agent = _agent_cache[user_id]
        
        # Clear history if requested
        if clear_history:
            agent.chat_memory.clear_history()
        
        # Get response
        response = agent.query(query)
        
        return response
        
    except Exception as e:
        print(f"Error in chat_service: {e}")
        return f"Error: {str(e)}"