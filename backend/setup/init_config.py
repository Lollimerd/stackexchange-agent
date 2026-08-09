"""Setting up ollama models, vectorstores and Neo4j Configs"""

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import SearchType
from typing import Dict
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
def answer_LLM():
    return ChatOllama(
        model="qwen3:1.7b",
        base_url=OLLAMA_BASE_URL,
        num_ctx=40968,  # 40k context
        num_predict=8192,  # max tokens in answer
        temperature=0.7,  # more creative
        repeat_penalty=1.5,  # higher, penalise repetitions
        repeat_last_n=-1,  # look back within context to penalise penalty
        top_p=0.5,  # more focused text
        top_k=10,  # give less diverse answers
        num_thread=8,
        reasoning=True,
    )


# embedding model
def embedding_model():
    return OllamaEmbeddings(
        model="jina/jina-embeddings-v2-base-en:latest",
        base_url=OLLAMA_BASE_URL,
        num_ctx=8192,  # 8k context
        num_thread=16,
    )


# reranker model
def reranker_model():
    return HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
        model_kwargs={"device": "cuda"},  # Use 'cuda' for GPU acceleration
    )


_graph_instance = None


def get_graph_instance() -> Neo4jGraph:
    """Get or create a reusable Neo4j graph instance (connection pooling)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = Neo4jGraph(
            url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
        )
    return _graph_instance


# print(f"\nschema: {graph.schema}\n")


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
            "text_node_properties": [
                "score",
                "link",
                "favourite_count",
                "id",
                "creation_date",
                "body",
                "title",
            ],
        },
        {
            "node_label": "Answer",
            "text_node_properties": [
                "score",
                "is_accepted",
                "id",
                "body",
                "creation_date",
            ],
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
            search_type=SearchType.HYBRID,
            text_node_properties=config["text_node_properties"],
            retrieval_query=retrieval_query,
        )
        print(f"Created vectorstore for {index_name} index")

    return vectorstores
