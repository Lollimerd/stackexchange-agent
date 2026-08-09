"""Setting up ollama models, vectorstores and Neo4j Configs"""

import os, json, time
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph, Neo4jVector
from typing_extensions import List, TypedDict
from typing import Dict, Optional
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ===========================================================================================================================================================
# Step 1: Load Configuration: Docker, Neo4j, Ollama, Langchain
# ===========================================================================================================================================================

load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

# qwen3:8b works for now with limited context of 40k, qwen3:30b works with 256k max
ANSWER_LLM = ChatOllama(
    model="qwen3:0.6b", # Ensure your model produces <think> tags
    base_url=OLLAMA_BASE_URL, 
    num_ctx=40960, # 40k
    num_predict=-2, # fill context
    tfs_z=2.0, # reduce impact of less probable tokens from output
    repeat_penalty=1.5, # higher, penalise repetitions
    repeat_last_n=-1, # look back within context to penalise penalty
    top_p=0.95, # more diverse text
    top_k=100, # give more diverse answers
    mirostat=2.0, # enable mirostat 2.0 sampling for controlling perplexity
    mirostat_tau=8.0, # output diversity
    mirostat_eta=0.2, # learning rate, responsiveness
    reasoning=True,
) 

# embedding model
EMBEDDINGS = OllamaEmbeddings(
    model="jina/jina-embeddings-v2-base-en:latest", 
    base_url=OLLAMA_BASE_URL, 
    num_ctx=8192, # 8k context
)

# reranker model
RERANKER_MODEL = HuggingFaceCrossEncoder(
    model_name='BAAI/bge-reranker-base',
    )  # Use 'cuda' if you have a GPU available.

# neo4j connection
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    enhanced_schema=True,
    refresh_schema=True
)
# print(f"\nschema: {graph.schema}\n")

# define state for application
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        chat_history: chat history
        context: list of documents
        answer: answer
    """
    question: str
    chat_history: List[Dict[str, str]]
    context: List[Document]
    answer: str

# test embedding generation
# sample_embedding = EMBEDDINGS.embed_query("Hello world")
# print(f"\nSample embedding dimension: {len(sample_embedding)}")

# ===========================================================================================================================================================
# Creation of vector index, vectorstores and fulltext for hybrid vector search
# ===========================================================================================================================================================

def create_vector_stores(graph, EMBEDDINGS, retrieval_query) -> Dict[str, Neo4jVector]:
    """
    Creates Neo4jVector stores from an existing graph using a data-driven approach.

    Args:
        graph: The Neo4j graph instance.
        EMBEDDINGS: The embedding model.
        retrieval_query: The Cypher query for retrieval.

    Returns:
        A dictionary of Neo4jVector store instances, keyed by their node label.
    """
    
    # Define a list of configurations for each vector store
    store_configs = [
        {
            "node_label": "Tag",
            "text_node_properties": ["name"],
        },
        {
            "node_label": "User",
            "text_node_properties": ["reputation", "display_name"],
        },
        {
            "node_label": "Question",
            "text_node_properties": ["score", "link", "favourite_count", "id", "creation_date", "body", "title"],
        },
        {
            "node_label": "Answer",
            "text_node_properties": ["score", "is_accepted", "id", "body", "creation_date"],
        },
    ]

    vectorstores = {}
    
    # Loop through the configurations and create the vectorstores & vector indexes
    for config in store_configs:
        label = config["node_label"]
        index_name = f"{label}_index"
        keyword_index_name = f"{label}_keyword_index"
        
        vectorstores[label.lower() + "store"] = Neo4jVector.from_existing_graph(
            graph=graph,
            node_label=label,
            embedding=EMBEDDINGS,
            embedding_node_property="embedding",
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type="hybrid",
            text_node_properties=config["text_node_properties"],
            retrieval_query=retrieval_query,
        )
        print(f"Created vectorstore for {index_name} index")
        
    return vectorstores