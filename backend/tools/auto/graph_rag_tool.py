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

from setup.init_config import cypher_LLM, embedding_model, get_graph_instance, reranker_model
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
            top_n=10,
        )
        logger.info("CrossEncoderReranker initialised (top_n=20).")
    return _compressor

CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a StackOverflow knowledge graph.

Instructions:
- Use ONLY the relationship types and node properties defined in the schema below.
- Output ONLY the raw Cypher statement — no explanation, no markdown, no apologies.
- ALWAYS start with a vector index search using `$question_embedding`. E.g.:
    CALL db.index.vector.queryNodes('Question_index', 50, $question_embedding) YIELD node AS q, score
- Follow the vector search with graph traversal anchored on Question-Answer pairs:
    MATCH (a:Answer)-[:ANSWERS]->(q:Question)
- Use OPTIONAL MATCH for supplementary context (tags, users) so that missing
  branches never eliminate otherwise valid results.
- When the question concerns a specific topic or technology, filter by shared
  community membership using:
    ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId)
  This ensures answers and questions belong to the same community cluster.
- Prefer accepted or high-scoring answers (a.is_accepted = true OR a.score > 0).
- Limit results (LIMIT 50 is a good default).
- Return enough fields for a rich answer: question title, question body, answer body,
  answer score, whether the answer is accepted, relevant tag names, and vector search similarity score (`score`).
- CRITICAL: In Cypher, if a query uses aggregation (such as `collect(DISTINCT...)`) in the `RETURN` clause, any variable used in the `ORDER BY` clause (like `score`) MUST be explicitly included in the `RETURN` clause. Make sure to project `score` in your `RETURN` statement if you order by it!

Schema:
{schema}

Examples:

# Find accepted answers for questions about Python async
CALL db.index.vector.queryNodes('Question_index', 50, $question_embedding) YIELD node AS q, score
MATCH (a:Answer)-[:ANSWERS]->(q:Question)
WHERE ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId)
  AND (a.is_accepted = true OR a.score > 0)
OPTIONAL MATCH (q)-[:TAGGED]->(t:Tag)
OPTIONAL MATCH (u:User)-[:PROVIDED]->(a)
RETURN q.title        AS question_title,
       q.body         AS question_body,
       a.body         AS answer_body,
       a.score        AS answer_score,
       collect(DISTINCT t.name) AS tags,
       u.display_name AS answered_by,
       score
ORDER BY score DESC, a.score DESC
LIMIT 25

# Top highest-scored questions with their best answer
CALL db.index.vector.queryNodes('Question_index', 50, $question_embedding) YIELD node AS q, score
MATCH (a:Answer)-[:ANSWERS]->(q)
WHERE ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId)
OPTIONAL MATCH (q)-[:TAGGED]->(t:Tag)
RETURN q.title        AS question_title,
       q.score        AS question_score,
       a.body         AS best_answer_body,
       a.score        AS answer_score,
       collect(DISTINCT t.name) AS tags,
       score
ORDER BY score DESC, q.score DESC
LIMIT 25

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
            verbose=False,                       # enable when debugging
            top_k=25,
        )
        logger.info("GraphCypherQAChain initialised (retrieval-only mode).")
    return _chain


# ---------------------------------------------------------------------------
# Internal helpers wrapped as named Runnables for SSE status visibility
# ---------------------------------------------------------------------------

def _run_graph_traversal(question: str) -> list[dict[str, Any]]:
    """
    Generate a Cypher query from *question* and execute it against Neo4j.
    Uses vector index search via $question_embedding for fast, indexed lookups.
    Returns the raw list of records returned by Neo4j.
    """
    chain = _get_chain()

    # Step 1: Embed the question — needed by the vector index in generated Cypher.
    question_embedding = embedding_model().embed_query(question)

    # Step 2: Generate Cypher using the cypher generation sub-chain only.
    # Passing question_embedding in the args allows the prompt template to
    # reference it; more importantly we pass it as a Bolt param in step 3.
    args = {
        "question": question,
        "schema": chain.graph_schema,
        "query": question,
    }
    from langchain_neo4j.chains.graph_qa.cypher import extract_cypher
    raw_cypher = chain.cypher_generation_chain.invoke(args)
    generated_cypher = extract_cypher(raw_cypher)

    if chain.cypher_query_corrector:
        generated_cypher = chain.cypher_query_corrector(generated_cypher)

    logger.info("Generated Cypher: %s", generated_cypher)

    # Step 3: Execute Cypher against Neo4j with $question_embedding as a
    # Bolt parameter so CALL db.index.vector.queryNodes(...) can resolve it.
    raw_context: list[dict[str, Any]] = []
    if generated_cypher:
        try:
            raw_context = chain.graph.query(
                generated_cypher,
                params={"question_embedding": question_embedding},
            )
        except Exception as exc:
            logger.error("Failed to execute generated Cypher: %s", exc)
            raise exc

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
    # Truncate heavy fields *before* string concatenation to avoid
    # allocating multi-MB page_content strings for documents the reranker will never fully read (cross-encoder max is ~512 tokens ≈ 2 000 chars)
    _MAX_BODY = 1500   # chars per field fed to the cross-encoder
    _MAX_META = 3500   # chars for metadata strings passed to the answer LLM
    docs: List[Document] = []
    for record in raw_records:
        title   = (record.get("question_title") or "")[:200]
        q_body  = (record.get("question_body") or "")[:_MAX_BODY]
        a_body  = (record.get("answer_body") or record.get("best_answer_body") or "")[:_MAX_BODY]
        page_content = f"Title: {title}\nQuestion: {q_body}\nAnswer: {a_body}"

        # Truncate remaining metadata string fields for the answer LLM context
        meta: Dict[str, Any] = {
            k: (v[:_MAX_META] if isinstance(v, str) and len(v) > _MAX_META else v)
            for k, v in record.items()
        }
        docs.append(Document(page_content=page_content, metadata=meta))

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
    Call it **at most once** per user message (or **twice** if the initial search did not provide good results).

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
