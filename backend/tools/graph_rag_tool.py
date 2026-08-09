from setup.init_config import (
    get_graph_instance,
    embedding_model,
    create_vector_stores,
    reranker_model,
    answer_LLM,
)
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List, Dict, Optional, Type, Any
from langchain_core.documents import Document
from prompts.system_prompts import analyst_prompt
from utils.util import format_docs_with_metadata, escape_lucene_chars
from langchain_core.tools import BaseTool
from agent.messages import format_chat_history
from middleware.langchain_middleware import (
    process_with_topic_analysis,
)
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from pydantic import BaseModel, Field
import logging


logger = logging.getLogger(__name__)

# ===========================================================================================================================================================
# Crafting custom cypher retrieval queries
# ===========================================================================================================================================================
import_query = """
    UNWIND $data AS q
    MERGE (question:Question {id:q.question_id}) 
    ON CREATE SET question.title = q.title, question.link = q.link, question.score = q.score,
        question.favorite_count = q.favorite_count, question.creation_date = datetime({epochSeconds: q.creation_date}),
        question.body = q.body_markdown, question.embedding = q.embedding
    FOREACH (tagName IN q.tags | 
        MERGE (tag:Tag {name:tagName}) 
        MERGE (question)-[:TAGGED]->(tag)
    )
    FOREACH (a IN q.answers |
        MERGE (question)<-[:ANSWERS]-(answer:Answer {id:a.answer_id})
        SET answer.is_accepted = a.is_accepted,
            answer.score = a.score,
            answer.creation_date = datetime({epochSeconds:a.creation_date}),
            answer.body = a.body_markdown,
            answer.embedding = a.embedding
        MERGE (answerer:User {id:coalesce(a.owner.user_id, "deleted")}) 
        ON CREATE SET answerer.display_name = a.owner.display_name,
                      answerer.reputation= a.owner.reputation
        MERGE (answer)<-[:PROVIDED]-(answerer)
    )
    WITH * WHERE NOT q.owner.user_id IS NULL
    MERGE (owner:User {id:q.owner.user_id})
    ON CREATE SET owner.display_name = q.owner.display_name,
                  owner.reputation = q.owner.reputation
    MERGE (owner)-[:ASKED]->(question)
    """

retrieval_query = """
// Start from vector search result variables: `node`, `score`
WITH node, score
// Route any node type to related Question(s) via UNION branches to avoid implicit grouping
CALL {
  WITH node
  // If node is a Question, use it directly
  WITH node
  MATCH (q:Question)
  WHERE node:Question AND elementId(q) = elementId(node)
  RETURN q
  UNION
  // If node is an Answer, route to its Question
  WITH node
  MATCH (node:Answer)-[:ANSWERS]->(q:Question)
  RETURN q
  UNION
  // If node is a Tag, route to Questions tagged with it
  WITH node
  MATCH (q:Question)-[:TAGGED]->(node:Tag)
  RETURN q
  UNION
  // If node is a User, include Questions they asked
  WITH node
  MATCH (node:User)-[:ASKED]->(q:Question)
  RETURN q
  UNION
  // If node is a User, include Questions they answered
  WITH node
  MATCH (node:User)-[:PROVIDED]->(:Answer)-[:ANSWERS]->(q:Question)
  RETURN q
}
WITH DISTINCT q AS question, node, score

// Community detection: compute overlap and optionally filter to same community when available
WITH
  question,
  node,
  score,
  any(x IN coalesce(question.CommunityId, []) WHERE x IN coalesce(node.CommunityId, [])) AS sameCommunity,
  (size(coalesce(question.CommunityId, [])) > 0 AND size(coalesce(node.CommunityId, [])) > 0) AS bothHaveCommunity
WHERE NOT bothHaveCommunity OR sameCommunity

// Build rich context for each question
// Core question data
WITH DISTINCT question, score, sameCommunity,
     coalesce(question.CommunityId, []) AS qComm,
     coalesce(node.CommunityId, []) AS nComm,
     {
  id: question.id,
  title: question.title,
  body: question.body,
  link: question.link,
  score: question.score,
  favorite_count: question.favorite_count,
  creation_date: toString(question.creation_date)
} AS questionDetails

// Askers
OPTIONAL MATCH (asker:User)-[:ASKED]->(question)
WITH question, score, sameCommunity, qComm, nComm, questionDetails, {
  id: asker.id,
  display_name: asker.display_name,
  reputation: asker.reputation
} AS askerDetails

// Tags
OPTIONAL MATCH (question)-[:TAGGED]->(tag:Tag)
WITH question, score, sameCommunity, qComm, nComm, questionDetails, askerDetails,
     COLLECT(DISTINCT tag.name) AS tags

// Answers + providers
OPTIONAL MATCH (answer:Answer)-[:ANSWERS]->(question)
OPTIONAL MATCH (provider:User)-[:PROVIDED]->(answer)
WITH question, score, sameCommunity, qComm, nComm, questionDetails, askerDetails, tags,
     COLLECT(DISTINCT {
       id: answer.id,
       body: answer.body,
       score: answer.score,
       is_accepted: answer.is_accepted,
       creation_date: toString(answer.creation_date),
       provided_by: {
         id: provider.id,
         display_name: provider.display_name,
         reputation: provider.reputation
       }
     }) AS answers

// Final projection
RETURN
  'Title: ' + coalesce(question.title, '') + '\\nBody: ' + coalesce(question.body, '') AS text,
  {
    question_details: questionDetails,
    asked_by: askerDetails,
    tags: tags,
    answers: {
      answers: answers
    },
    community: {
      questionCommunityId: qComm,
      nodeCommunityId: nComm,
      sameCommunity: sameCommunity
    },
    simscore: score
  } AS metadata,
  score
ORDER BY score DESC
LIMIT 50
"""

# Create vector stores with error handling
try:
    stores = create_vector_stores(
        get_graph_instance(), embedding_model(), retrieval_query
    )
    tagstore = stores.get("tagstore")
    userstore = stores.get("userstore")
    questionstore = stores.get("questionstore")
    answerstore = stores.get("answerstore")

    # Verify all stores were created
    if not all([tagstore, userstore, questionstore, answerstore]):
        logger.warning("Some vector stores were not created successfully")
except Exception as e:
    logger.error(f"Error creating vector stores: {e}")
    raise

# create compressor
try:
    compressor = CrossEncoderReranker(
        model=reranker_model(),
        top_n=10,  # This will return the top n most relevant documents.
    )
except Exception as e:
    logger.error(f"Error creating compressor: {e}")
    raise

# ===========================================================================================================================================================
# Setting Up Retrievers from vectorstores for EnsembleRetriever
# ===========================================================================================================================================================


# Split retrieval into steps for observability
def retrieve_raw_docs(question: str) -> List[Document]:
    """Step 1: Graph Traversal & Ensemble Retrieval"""
    try:
        # Define the common search arguments once
        common_search_kwargs = {
            "k": 50,  # Increased initial pool: wider net across all entity types
            "score_threshold": 0.9,  # Slightly lowered to ensure we catch cross-domain links
            "fetch_k": 10000,  # Number of candidates for the initial vector search
            "lambda_mult": 0.5,  # Balanced weight between Vector and Full-text
            "params": {
                "embedding": embedding_model().embed_query(question),
                "keyword_query": escape_lucene_chars(question),
            },
        }

        # Use a list of vectorstores, filtering out any that failed to initialize
        vectorstores = [
            s
            for s in [tagstore, userstore, questionstore, answerstore]
            if s is not None
        ]

        if not vectorstores:
            logger.warning("No vector stores available for retrieval")
            return []

        # Create the retrievers using a list comprehension
        retrievers = [
            store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=common_search_kwargs,
            )
            for store in vectorstores
        ]

        # Calculate equal weights dynamically
        num_retrievers = len(retrievers)
        weights = [1.0 / num_retrievers] * num_retrievers

        # init ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights,
        )

        logger.info(f"--- 🌐 GLOBAL RETRIEVAL: {question} ---")
        docs = ensemble_retriever.invoke(question)
        logger.info(f"Graph Traversal Complete. Found {len(docs)} documents.")
        return docs
    except Exception as e:
        logger.error(f"Error in retrieve_raw_docs: {e}")
        return []


def rerank_docs(inputs: Dict) -> List[Document]:
    """Step 2: Reranking"""
    try:
        docs = inputs.get("docs", [])
        question = inputs.get("question", "")

        RELEVANCY_THRESHOLD = 0.95

        if not docs:
            return []

        logger.info(f"Reranking {len(docs)} documents...")
        reranked_docs = compressor.compress_documents(documents=docs, query=question)

        # ✨ RELEVANCE GUARDRAIL: Filter by score
        high_quality_docs = [
            doc
            for doc in reranked_docs
            if doc.metadata.get("relevance_score", 1.0) >= RELEVANCY_THRESHOLD
        ]

        # Handle Low-Confidence Situations
        if not high_quality_docs:
            logger.warning(
                f"⚠️ GUARDRAIL TRIGGERED: No docs met threshold {RELEVANCY_THRESHOLD}"
            )
            return []  # Returns empty context to trigger LLM fallback

        logger.info(f"✅ {len(high_quality_docs)} docs passed relevancy guardrail.")
        return high_quality_docs
    except Exception as e:
        logger.error(f"Error in rerank_docs: {e}")
        return []


# ===========================================================================================================================================================
# Chain Assembly
# ===========================================================================================================================================================

# 1. Retrieval Sequence: Fetch -> Rerank
retrieval_chain = RunnablePassthrough.assign(
    docs=lambda x: RunnableLambda(retrieve_raw_docs)
    .with_config(run_name="GraphTraversal")
    .invoke(x["question"])
) | RunnableLambda(rerank_docs).with_config(run_name="Reranking")

# 2. Main GraphRAG Chain
# Flow: Input -> Context/History Prep -> Topic Analysis -> LLM Generation
try:
    # Prepare inputs for the LLM (Context + History)
    input_preparation = RunnablePassthrough.assign(
        context=retrieval_chain | format_docs_with_metadata,
        chat_history_formatted=lambda x: format_chat_history(x.get("chat_history", [])),
    )

    graph_rag_chain = (
        input_preparation
        # Topic Analysis Middleware: Inspects context/history to maintain session topic
        | process_with_topic_analysis
        # inject system prompt
        | analyst_prompt
    )

    logger.info("GraphRAG chain with topic analysis initialized successfully")
except Exception as e:
    logger.error(f"Error initializing GraphRAG chain: {e}")
    raise


# define state for application
class GraphRAGInput(BaseModel):
    """Input for the GraphRAG tool."""

    question: str = Field(description="The question to ask the knowledge base")
    # Make other fields optional and loose to prevent LLM validation errors
    # The LLM likely won't provide these anyway, so defaults work.
    chat_history: Optional[Any] = Field(
        description="Previous chat history context (optional)", default=[]
    )
    session_topic: Optional[str] = Field(
        description="Current session topic for continuity (optional)", default=""
    )
    session_id: Optional[str] = Field(
        description="Unique session identifier (optional)", default=""
    )


class GraphRAGTool(BaseTool):
    """Tool that queries the GraphRAG knowledge base."""

    name: str = "graph_rag_tool"
    description: str = "ALWAYS use this tool for ANY technical question, code snippet, or knowledge base query. Do not answer from memory."
    args_schema: Type[BaseModel] = GraphRAGInput

    # synchronous execution
    def _run(
        self,
        question: str,
        chat_history: Optional[Any] = [],
        session_topic: str = "",
        session_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool synchronously."""
        # Ensure chat_history is a list if it comes as None or something else
        if chat_history is None:
            chat_history = []
        if not isinstance(chat_history, list):
            # Fallback if LLM sends a string or other type
            chat_history = []
        result = graph_rag_chain.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "session_topic": session_topic,
                "session_id": session_id,
            },
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )
        return result.to_string()

    # asynchronous execution
    async def _arun(
        self,
        question: str,
        chat_history: Optional[Any] = [],
        session_topic: str = "",
        session_id: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool asynchronously."""
        # Ensure chat_history is a list
        if chat_history is None:
            chat_history = []
        if not isinstance(chat_history, list):
            chat_history = []
        result = await graph_rag_chain.ainvoke(
            {
                "question": question,
                "chat_history": chat_history,
                "session_topic": session_topic,
                "session_id": session_id,
            },
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )
        return result.to_string()


# Initialize the tool
graph_rag_tool = GraphRAGTool()
