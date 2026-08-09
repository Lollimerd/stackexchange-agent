from setup.init import (
    graph,
    EMBEDDINGS,
    create_vector_stores,
    ANSWER_LLM,
    RERANKER_MODEL,
)

from langchain_classic.retrievers.ensemble import EnsembleRetriever

from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List, Dict, Optional, Type
from langchain_core.documents import Document
from prompts.st_overflow import analyst_prompt
from utils.util import format_docs_with_metadata, escape_lucene_chars
from langchain_core.tools import BaseTool
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
    stores = create_vector_stores(graph, EMBEDDINGS, retrieval_query)
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
        model=RERANKER_MODEL,
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
                "embedding": EMBEDDINGS.embed_query(question),
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


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history for inclusion in the prompt."""
    try:
        if not chat_history:
            return ""

        formatted_history = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted_history.append(f"User: {content}")
            elif role == "assistant":
                # Only include the main content, not the thought process
                formatted_history.append(f"Assistant: {content}")

        return "\n".join(formatted_history)
    except Exception as e:
        logger.error(f"Error formatting chat history: {e}")
        return ""


def check_context_presence(input_dict: Dict) -> Dict:
    """Adds a system flag if the guardrail blocked all documents."""
    context = input_dict.get("context", "")
    # Check if our specific content delimiter is present
    if not context or "--------- CONTENT ---------" not in context:
        input_dict["context"] = (
            "[SYSTEM NOTE: NO RELEVANT DATA FOUND IN KNOWLEDGE GRAPH]"
        )
        logger.info("Context fallback applied: No relevant data found.")
    return input_dict


def process_with_topic_analysis(input_dict: Dict) -> Dict:
    """Enriches input with topic continuity analysis before processing."""
    try:
        from utils.topic_manager import TopicManager

        question = input_dict.get("question", "")
        session_topic = input_dict.get("session_topic", "")
        session_id = input_dict.get("session_id", "")

        # Calculate similarity
        similarity_data = TopicManager.calculate_topic_similarity(
            question, session_topic
        )

        # Get relevant previous context
        relevant_context = []
        if session_id:
            relevant_context = TopicManager.get_relevant_context_for_continuation(
                session_id, question, max_messages=3
            )

        # Format relevant context
        relevant_context_str = ""
        if relevant_context:
            for msg in relevant_context:
                role = "User" if msg["role"] == "user" else "Assistant"
                relevant_context_str += f"{role}: {msg['content']}\n\n"

        # Return enriched input
        return {
            **input_dict,
            "session_topic": session_topic or "General Discussion",
            "topic_similarity_score": f"{similarity_data['similarity_score']:.2f}",
            "topic_confidence": similarity_data["confidence_level"],
            "topic_status": similarity_data["recommendation"],
            "relevant_context": relevant_context_str
            or "[No previous context available]",
            "continuity_instruction": similarity_data["recommendation"],
        }
    except Exception as e:
        logger.error(f"Error in topic analysis: {e}")
        return {
            **input_dict,
            "session_topic": input_dict.get("session_topic", "General Discussion"),
            "topic_similarity_score": "0.50",
            "topic_confidence": "low",
            "topic_status": "Unable to determine topic status",
            "relevant_context": "[Error retrieving relevant context]",
            "continuity_instruction": "Proceed with caution due to analysis error",
        }


retrieval_chain = RunnablePassthrough.assign(
    docs=lambda x: RunnableLambda(retrieve_raw_docs)
    .with_config(run_name="GraphTraversal")
    .invoke(x["question"])
) | RunnableLambda(rerank_docs).with_config(run_name="Reranking")

# This chain is activated for GraphRAG.
try:
    graph_rag_chain = (
        RunnablePassthrough.assign(
            context=retrieval_chain | format_docs_with_metadata,
            chat_history_formatted=lambda x: format_chat_history(
                x.get("chat_history", [])
            ),
        )
        # Fix: Run topic analysis ONCE and merge results
        | process_with_topic_analysis
        | analyst_prompt
        | ANSWER_LLM
    )
    logger.info("GraphRAG chain with topic analysis initialized successfully")
except Exception as e:
    logger.error(f"Error initializing GraphRAG chain: {e}")
    raise


class GraphRAGInput(BaseModel):
    """Input for the GraphRAG tool."""

    question: str = Field(description="The question to ask the knowledge base")
    chat_history: List[Dict] = Field(
        description="Previous chat history context", default=[]
    )
    session_topic: str = Field(
        description="Current session topic for continuity", default=""
    )
    session_id: str = Field(description="Unique session identifier", default="")


class GraphRAGTool(BaseTool):
    """Tool that queries the GraphRAG knowledge base."""

    name: str = "graph_rag_tool"
    description: str = "Retrieves information from the graph database to answer questions with topic continuity analysis."
    args_schema: Type[BaseModel] = GraphRAGInput

    def _run(
        self,
        question: str,
        chat_history: List[Dict] = [],
        session_topic: str = "",
        session_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool synchronously."""
        result = graph_rag_chain.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "session_topic": session_topic,
                "session_id": session_id,
            },
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )
        return str(result.content)

    async def _arun(
        self,
        question: str,
        chat_history: List[Dict] = [],
        session_topic: str = "",
        session_id: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool asynchronously."""
        result = await graph_rag_chain.ainvoke(
            {
                "question": question,
                "chat_history": chat_history,
                "session_topic": session_topic,
                "session_id": session_id,
            },
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )
        return str(result.content)


# Initialize the tool
graph_rag_tool = GraphRAGTool()
