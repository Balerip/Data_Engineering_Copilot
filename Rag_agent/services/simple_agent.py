import os
import asyncio
from pathlib import Path
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from index import Index
from chat_memory import ChatMemory

class SimpleWorkingAgent:
    def __init__(self, directory: str, storage_directory: str, user_id: str):
        self.directory = directory
        self.storage_directory = storage_directory
        self.user_id = user_id
        # Initialize chat memory - we'll handle serialization carefully
        try:
            self.chat_memory = ChatMemory(user_id=user_id)
        except:
            print("Note: Chat memory initialization failed, continuing without it")
            self.chat_memory = None
        
        # Simple URLs that definitely have the content we need
        self.urls = [
            "https://spark.apache.org/docs/latest/sql-getting-started.html",
            "https://spark.apache.org/docs/latest/sql-programming-guide.html",
            "https://docs.getdbt.com/reference/references-overview",
            "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html"
        ]
        
        print("Loading index...")
        self.index = Index(directory, storage_directory).load_index(urls=self.urls)
        
        if not self.index:
            raise ValueError("Failed to create index")
        
        # Setup models
        print("Setting up models...")
        embedding = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"attn_implementation": "eager"}
        )
        
        # CRITICAL: Set temperature to 0 for deterministic responses
        llm = Ollama(model="llama3", request_timeout=120.0, temperature=0.0)
        
        Settings.embed_model = embedding
        Settings.llm = llm
        
        # Create working query engine with stricter settings
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )
        
        # Create tool with very explicit description
        tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=ToolMetadata(
                name="docs_search",
                description="Search ONLY in indexed Apache Spark, dbt, and Apache Airflow documentation. This tool contains NO Kafka, NO Snowflake, NO other technology docs."
            )
        )
        
        # MUCH STRICTER system prompt
        system_prompt = """You are a documentation assistant with access to ONLY these specific documents:
        1. Apache Spark SQL documentation
        2. dbt (data build tool) documentation  
        3. Apache Airflow DAGs documentation

        CRITICAL RULES:
        - You MUST use the docs_search tool for EVERY answer
        - You can ONLY answer based on what the docs_search tool returns
        - If docs_search returns no relevant information, say: "I don't have documentation for that topic. I only have Spark, dbt, and Airflow docs."
        - NEVER use your general knowledge about Kafka, Snowflake, or any other technology
        - NEVER answer questions about topics not in the indexed documents
        - If asked about Kafka specifically, respond: "I don't have Kafka documentation. I can only help with Spark, dbt, and Airflow."
        - Always cite which document your answer comes from (Spark/dbt/Airflow)
        
        Before answering ANY question:
        1. Use docs_search tool FIRST
        2. Check if the returned content is actually relevant
        3. If not relevant, admit you don't have that information
        4. NEVER fill in gaps with general knowledge"""
        
        # Create ReActAgent with strict settings
        self.agent = ReActAgent(
            tools=[tool],
            llm=llm,
            verbose=True,
            max_iterations=2,  # Reduce iterations to prevent fallback to general knowledge
            context=system_prompt,
            # Add these to make it stricter
            handle_parsing_errors=False,
            early_stopping_method="force"  # Stop early if no good answer from tools
        )
        
        print("ReActAgent ready with STRICT document-only mode!")
    
    def save_to_memory(self, question: str, answer: str):
        """Save Q&A to memory, handling serialization issues"""
        if self.chat_memory:
            try:
                # Convert to simple strings to avoid serialization issues
                self.chat_memory.put_messages({
                    "question": str(question),
                    "answer": str(answer)
                })
            except Exception as e:
                print(f"Note: Could not save to chat memory: {e}")
    
    def query(self, question: str) -> str:
        """
        Simple sync wrapper for async ReActAgent
        """
        # Pre-check for explicitly unsupported topics
        unsupported_topics = ['kafka', 'snowflake', 'mongodb', 'redis', 'elasticsearch', 'postgres', 'mysql']
        question_lower = question.lower()
        
        for topic in unsupported_topics:
            if topic in question_lower:
                return f"I don't have {topic.capitalize()} documentation. I can only help with Apache Spark, dbt, and Apache Airflow based on the documents I have indexed."
        
        async def run_agent():
            try:
                print(f"\n=== Processing: {question} ===")
                
                # The workflow ReActAgent expects user_msg parameter
                response = await self.agent.run(user_msg=question)
                
                # Extract response text - handle different response types
                if hasattr(response, 'response'):
                    result = str(response.response)
                elif hasattr(response, 'output'):
                    result = str(response.output)
                elif hasattr(response, 'content'):
                    result = str(response.content)
                else:
                    result = str(response)
                
                # Final check - if the response mentions technologies we don't have docs for
                if any(topic in result.lower() for topic in unsupported_topics):
                    return "I can only provide information from the Spark, dbt, and Airflow documentation I have indexed. I cannot answer about other technologies."
                
                return result
                    
            except Exception as e:
                print(f"Agent error: {e}")
                # Don't fallback to general knowledge
                return "I encountered an error searching the documentation. Please rephrase your question about Spark, dbt, or Airflow."
        
        # Run the async function synchronously
        result = asyncio.run(run_agent())
        
        # Save to chat memory (handling serialization)
        self.save_to_memory(question, result)
        
        return result


def main():
    """Simple interactive chat"""
    # Create directories
    os.makedirs("./dataStorage", exist_ok=True)
    os.makedirs("./storage", exist_ok=True)
    
    print("Initializing agent...")
    agent = SimpleWorkingAgent(
        directory="./dataStorage",
        storage_directory="./storage",
        user_id="user123"
    )
    
    print("\n=== Agent Ready! ===")
    print("Available documentation: Apache Spark, dbt, Apache Airflow ONLY")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("\nYou: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not question:
            continue
            
        answer = agent.query(question)
        print(f"\nAgent: {answer}")


# Quick test function
def test_agent():
    """Test the agent with a simple question"""
    agent = SimpleWorkingAgent(
        directory="./dataStorage",
        storage_directory="./storage",
        user_id="test_user"
    )
    
    # Test questions - including one that should be rejected
    questions = [
        "What is Apache Spark?",
        "Give me 3 quiz questions about Airflow DAGs",
        "Tell me about Kafka streams"  # This should be rejected
    ]
    
    for q in questions:
        print(f"\nTesting: {q}")
        answer = agent.query(q)
        print(f"Answer: {answer[:200]}...")  # Print first 200 chars


if __name__ == "__main__":
    # Run interactive chat
    main()
    
    # Or run test
    # test_agent()