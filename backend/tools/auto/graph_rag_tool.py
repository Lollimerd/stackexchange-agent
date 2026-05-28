"""
graph_rag_tool — Graph-RAG retrieval tool for the StackExchange agent.

Pipeline
--------
  1. GraphTraversal  — cypher_LLM generates a Cypher query; it is executed
                       against Neo4j; raw dict records are returned.
  2. Reranking       — raw dicts are wrapped into Document objects and fed
                       through CrossEncoderReranker (BAAI/bge-reranker-base)
                       which scores each doc against the question and keeps
                       the top-N most relevant ones.
  3. Formatting      — format_docs_with_metadata() serialises the reranked
                       Documents into a readable context string for answer_LLM.

Streaming hooks
---------------
Both steps are wrapped as named RunnableLambdas so that backend.py can emit
SSE status events for the UI:
  • "GraphTraversal"  — fires while Cypher is being generated and executed.
  • "Reranking"       — fires while the cross-encoder scores documents.
"""

import logging
from typing import Any, Dict, List

from langchain.tools import tool
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_neo4j import GraphCypherQAChain

from setup.init_config import cypher_LLM, get_graph_instance, reranker_model
from utils.util import format_docs_with_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-initialised singletons
# ---------------------------------------------------------------------------
_chain: GraphCypherQAChain | None = None
_compressor: CrossEncoderReranker | None = None


def _get_compressor() -> CrossEncoderReranker:
    """Build (or return cached) CrossEncoderReranker."""
    global _compressor
    if _compressor is None:
        _compressor = CrossEncoderReranker(
            model=reranker_model(),
            top_n=20,
        )
        logger.info("CrossEncoderReranker initialised (top_n=20).")
    return _compressor

CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a StackOverflow knowledge graph.

Instructions:
- Use ONLY the relationship types and node properties defined in the schema below.
- Output ONLY the raw Cypher statement — no explanation, no markdown, no apologies.
- ALWAYS anchor the query on Question-Answer pairs using the pattern:
    (a:Answer)-[:ANSWERS]->(q:Question)
- Use OPTIONAL MATCH for supplementary context (tags, users) so that missing
  branches never eliminate otherwise valid results.
- When the question concerns a specific topic or technology, filter by shared
  community membership using:
    ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId)
  This ensures answers and questions belong to the same community cluster.
- Prefer accepted or high-scoring answers (a.is_accepted = true OR a.score > 0).
- Limit results to a reasonable number (LIMIT 100 unless the question asks for more).
- Return enough fields for a rich answer: question title, question body, answer body,
  answer score, whether the answer is accepted, and relevant tag names.

Schema:
{schema}

Examples:

# Find accepted answers for questions about Python async
MATCH (a:Answer)-[:ANSWERS]->(q:Question)
WHERE (toLower(q.title) CONTAINS 'async' OR toLower(q.body) CONTAINS 'async')
  AND ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId)
  AND (a.is_accepted = true OR a.score > 0)
OPTIONAL MATCH (q)-[:TAGGED]->(t:Tag)
OPTIONAL MATCH (u:User)-[:PROVIDED]->(a)
RETURN q.title        AS question_title,
       q.body         AS question_body,
       a.body         AS answer_body,
       a.score        AS answer_score,
       collect(DISTINCT t.name) AS tags,
       u.display_name AS answered_by
ORDER BY a.score DESC
LIMIT 100

# Top 5 highest-scored questions with their best answer
MATCH (a:Answer)-[:ANSWERS]->(q:Question)
  AND ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId)
OPTIONAL MATCH (q)-[:TAGGED]->(t:Tag)
RETURN q.title        AS question_title,
       q.score        AS question_score,
       a.body         AS best_answer_body,
       a.score        AS answer_score,
       collect(DISTINCT t.name) AS tags
ORDER BY q.score DESC
LIMIT 100

The question is:
{question}"""


def _get_chain() -> GraphCypherQAChain:
    """Build (or return cached) GraphCypherQAChain configured for retrieval only."""
    global _chain
    if _chain is None:
        graph = get_graph_instance()
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_TEMPLATE,
        )
        _chain = GraphCypherQAChain.from_llm(
            llm=cypher_LLM(),                    # LLM for Cypher generation only
            graph=graph,
            cypher_prompt=cypher_prompt,         # inject domain-specific prompt
            validate_cypher=True,                # reject syntactically invalid queries
            return_intermediate_steps=True,      # exposes generated_cypher + context
            allow_dangerous_requests=True,       # required by langchain-neo4j ≥ 0.3
            verbose=False,
            top_k=100,                           # allow more docs to reach reranker
        )
        logger.info("GraphCypherQAChain initialised (retrieval-only mode).")
    return _chain


# ---------------------------------------------------------------------------
# Internal helpers wrapped as named Runnables for SSE status visibility
# ---------------------------------------------------------------------------

def _run_graph_traversal(question: str) -> list[dict[str, Any]]:
    """
    Generate a Cypher query from *question* and execute it.
    Returns the raw list of records returned by Neo4j.
    """
    chain = _get_chain()

    # invoke() always calls the QA step internally, but we only care about
    # intermediate_steps which contains the raw context.
    result = chain.invoke({"query": question})

    intermediate = result.get("intermediate_steps", [])
    raw_context: list[dict[str, Any]] = []

    for step in intermediate:
        # Each intermediate step is a dict that may contain:
        #   {"query": <cypher string>}   — the generated Cypher
        #   {"context": [<records>]}     — rows returned by Neo4j
        if "context" in step:
            raw_context = step["context"]
            break

    generated_cypher = ""
    for step in intermediate:
        if "query" in step:
            generated_cypher = step["query"]
            break

    logger.info("Generated Cypher: %s", generated_cypher)
    logger.info("Raw context records: %d rows", len(raw_context))

    return raw_context


# Wrap as a named RunnableLambda so backend.py can detect "GraphTraversal" events.
GraphTraversal = RunnableLambda(_run_graph_traversal).with_config(
    {"run_name": "GraphTraversal"}
)


def _rerank_docs(inputs: Dict[str, Any]) -> List[Document]:
    """
    Cross-encoder reranking step.

    Expects ``inputs`` to be a dict with:
      • ``"docs"``     — list[dict] raw records from Neo4j
      • ``"question"`` — the original user question string

    Each raw dict is converted to a Document whose page_content is a
    human-readable title+body string and whose metadata holds the full
    record.  CrossEncoderReranker then scores every Document against the
    question and returns the top-N most relevant ones.
    """
    raw_records: List[Dict[str, Any]] = inputs.get("docs", [])
    question: str = inputs.get("question", "")

    if not raw_records:
        logger.warning("Reranking received 0 documents — skipping.")
        return []

    # --- Convert raw dicts → Documents ---
    docs: List[Document] = []
    for record in raw_records:
        title = record.get("question_title", "")
        q_body = record.get("question_body", "")
        a_body = record.get("answer_body", record.get("best_answer_body", ""))
        page_content = f"Title: {title}\nQuestion: {q_body}\nAnswer: {a_body}"
        docs.append(Document(page_content=page_content, metadata=record))

    logger.info("Reranking %d documents...", len(docs))
    compressor = _get_compressor()

    try:
        reranked = compressor.compress_documents(documents=docs, query=question)
    except Exception as exc:
        logger.error("CrossEncoderReranker failed: %s — returning raw docs", exc)
        reranked = docs  # graceful fallback: return unranked docs

    final_docs = list(reranked)
    logger.info("✅ %d docs passed reranking.", len(final_docs))
    return final_docs


# Named Runnable so backend.py can detect "Reranking" events.
Reranking = RunnableLambda(_rerank_docs).with_config({"run_name": "Reranking"})


# ---------------------------------------------------------------------------
# Public LangChain tool
# ---------------------------------------------------------------------------

@tool
def graph_rag_tool(question: str) -> str:
    """
    Retrieve relevant information from the StackOverflow knowledge graph.

    Use this tool whenever the user asks a technical question about software,
    code, errors, or any topic that may be answered from the knowledge base.
    Call it **at most once** per user message.

    Args:
        question: The user's question or topic to look up in the graph.

    Returns:
        A JSON-formatted string containing the raw records retrieved from
        Neo4j.  The agent should use this data to compose its final answer.
    """
    logger.info("graph_rag_tool invoked: %r", question[:120])

    try:
        # Step 1 — Cypher generation + Neo4j execution
        raw_records: List[Dict[str, Any]] = GraphTraversal.invoke(question)

        # Step 2 — Cross-encoder reranking
        reranked_docs: List[Document] = Reranking.invoke(
            {"docs": raw_records, "question": question}
        )

        if not reranked_docs:
            return (
                "No relevant data found in the knowledge graph for this question. "
                "Answer using your general knowledge."
            )

        # Step 3 — Format reranked Documents into a context string
        return format_docs_with_metadata(reranked_docs)

    except Exception as exc:
        logger.error("graph_rag_tool error: %s", exc, exc_info=True)
        return f"Graph retrieval failed: {exc}. Answer using your general knowledge."
