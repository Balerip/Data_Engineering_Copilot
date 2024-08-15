import os
from fastapi import FastAPI, HTTPException
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from . import index
from dotenv import load_dotenv
from llama_index.embeddings.google import GooglePaLMEmbedding

# Load environment variables from .env file
load_dotenv()

class Agent:
    def __init__(self, directory: str, storage_directory: str,urls: list[str] = None):
        self.directory = directory
        self.storage_directory = storage_directory
        self.urls = ["https://studentfinance.northeastern.edu/"]
        print(f"Agent class, Data storage directory: {self.directory}")
        print(f"Agent class, Index storage directory: {self.storage_directory}")

        self.index = index.Index(directory, storage_directory).load_index(urls=self.urls)
        
        if self.index is None:
            raise ValueError("Failed to load or create the index.")
        
        print("Creating embedding model...")
        self.embedding_name =  "models/embedding-gecko-001"
        self.api_key="AIzaSyAchKo1r7xttfjHPCpaVpGzd7RyfqbvRdU"
        self.embedding=GooglePaLMEmbedding(model_name=self.embedding_name, api_key=self.api_key)

        print("Creating LLM model...")
        llm_model = Ollama(model="llama3", request_timeout=60.0)

        # Initialize Settings with the models
        Settings.embed_model = self.embedding
        Settings.llm = llm_model

        # Initialize query engine
        print("Creating query engine...")
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=2,
            embedding_model=Settings.embed_model,
            llm=Settings.llm  # Pass LLM model to the query engine
        )

        # Define metadata for the tool
        self.metadata = ToolMetadata(
            name="QueryEngineTool",
            description=("Handles Queries relsted to resources in Northeastern University Silicon Valley Campus")
        )

        # Initialize the query engine tool with metadata
        self.query_engine_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=self.metadata,
           
        )

        # Define ReAct Agent with the query engine tool
        self.query_agent = ReActAgent.from_tools([self.query_engine_tool], llm=Settings.llm, verbose=True,max_iterations=5)
        print("Agent created successfully.")

    def get_query_react_agent(self) -> ReActAgent:
        return self.query_agent
    
    def create_prompt(self,query: str) -> str:
        return (
        # "CONTEXT:\n"
        # "I am NUGPT, a helpful, fun, and friendly chat assistant for Northeastern University. "
        # "My primary function is to provide detailed information about the resources available at Northeastern University's Silicon Valley campus, "
        # "including courses and details for various programs.\n\n"
        # "OBJECTIVE:\n"
        # "Your task is to accurately and efficiently answer questions related to resources at Northeastern University's Silicon Valley campus, "
        # "with a focus on providing detailed and relevant information specific to the program mentioned in the query. Ensure that the responses are tailored to the program or resource in question. "
        # "For questions not related to campus resources, respond appropriately.\n\n"
        # "STYLE:\n"
        # "Write in an informative and instructional style, similar to an academic advisor. Ensure clarity and coherence in presenting each resource's information, making it easy for users to understand and use the information provided.\n\n"
        # "TONE:\n"
        # "Maintain a positive, motivational, and approachable tone throughout. Your responses should feel supportive and engaging, like a knowledgeable peer or advisor offering valuable insights and guidance.\n\n"
        # "AUDIENCE:\n"
        # "The target audience is current students at Northeastern University's Silicon Valley campus. Assume a readership that is looking for detailed, accurate, and practical information about resources for various programs to guide their decisions and actions.\n\n"
        # "RESPONSE FORMAT:\n"
        # "1. For questions about a specific program mentioned in the query, provide detailed and relevant information specific to that program.\n  For e.g: If asked about course from Information systems search only the Information systems file.\n Similarly do for the other programs too"  
        # "2. If you don't know the answer, respond with: 'I don't know.'\n"
        # "3. For questions not related to campus resources, respond with: 'I only answer questions about Northeastern Silicon Valley campus resources.'\n\n"
        # "QUERY:\n"
       "When building an LLM for specific data, such as course data, using expert prompting techniques can significantly improve the accuracy and relevance of the model's responses. Expert prompting involves crafting system prompts that guide the model towards more accurate, context-aware, and domain-specific outputs.\n"
       "CONTEXT:\n" 
       "Your LLM needs to respond to queries related to course offerings, prerequisites, instructors, and schedules for Northeastern University."
       "Purpose:\n"
       "Guide the model to provide precise, contextually relevant information on courses."
       "Structure:\n"
       "Specify the role of the LLM."
       "Define the types of questions it should answer."
       "Highlight any special considerations or constraints (e.g., prerequisites, course level)."
       "You are a knowledgeable academic advisor for Northeastern University. You have comprehensive data on all course offerings, including course descriptions, prerequisites, instructors, schedules, and enrollment capacities. Your goal is to assist students by providing accurate, detailed, and relevant answers to their questions about courses."
       "When a student asks about a course, ensure you:"
       "1. Confirm the course details, such as the title, code, and a brief description."
       "2. Provide prerequisite information, if any."
       "3. Mention the instructor's name and office hours."
       "4. Offer information about the schedule, including days and times of the week the course is offered."
       "5. If applicable, inform the student of any special enrollment considerations, such as waitlist options or departmental consent requirements."
       "6. If a student's question is ambiguous, ask for clarification before providing an answer."
       "7. Always prioritize accuracy and relevancy in your responses."
       "Your responses should be concise, direct, and free of unnecessary details. If you do not have enough information to answer a question, respond with a polite request for more information or suggest where the student can find the details."
       "For example, if a student asks about INFO 5001 course your response should include the course title, prerequisites, instructor, schedule, and any relevant enrollment details."
       "User Prompt:\n"
       "What is the description for INFO 5001?"
       "LLM Response:\n" 
       """The Description for INFO 5001 Application Modeling and Design course is that it practices social-technical 
       software engineering methods and tools to solve real-world problems. Explores innovative design and programming 
       techniques to build significant business applications quickly. Studies the process of systematically combining 
       UX techniques, business processes, and complex data models to assemble
       applications that are user-friendly and meet business requirements. Employs the object-oriented paradigm,
       visual user interface design principles, and programming languages such as Java, as well as productivity tools,
       to put together complicated, powerful business applications with ease. Explores the art of how to
       systematically write software programs to solve any business problem, through practicing simple and smart
       ways of making software programming enjoyable."""
       "User Prompt:\n"
       "Can I enroll in CS 6140 Machine Learning course? If yes, what are the prerequisites that I need to complete?"
       "LLM Response:\n"
       """The prerequisite courses for the above course is CS 5800 with a minimum grade of C- 
          or CS 7800 with a minimum grade of CCS 6200. Information Retrieval which provides an introduction to information retrieval systems and different approaches to information retrieval.
          Topics covered include evaluation of information retrieval systems; retrieval, language, and indexing models;
          file organization; compression; relevance feedback; clustering; distributed retrieval and metasearch;
          probabilistic approaches to information retrieval; Web retrieval; filtering, collaborative filtering, and
          recommendation systems; cross-language IR; multimedia IR; and machine learning for information retrieval."""
       "Query:\n"
       f"{query}"
        )
  

def get_query_response(message: str) -> str:
    """
    Processes a chat message and returns a response.
    """
    try:
        directory = os.environ.get("DATA_STORAGE_DIRECTORY")
        storage_directory = os.environ.get("INDEX_STORAGE_DIRECTORY")

        agent = Agent(directory, storage_directory)
        react_agent = agent.get_query_react_agent()
        custom_prompt = agent.create_prompt(message)
        query_response = react_agent.chat(custom_prompt)

        if hasattr(query_response, 'response'):
            agent_response = query_response.response
        else:
            agent_response = "Response attribute missing"

        return f"Agent response: {agent_response}"  # Returning actual response
    except Exception as e:
        # More detailed logging for debugging
        print(f"Error in get_query_response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the test
question_1 = "What is the course title and course description for course code DS 3000 in Data science program?"
print(get_query_response(question_1))

# question_2="What are the corequisites for the Data Science Programming Practicum course in Data Science program?"
# print(get_query_response(question_2))
