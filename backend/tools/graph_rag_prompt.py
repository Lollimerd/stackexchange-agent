from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)

# Define the System Message, which sets the AI's persona and instructions.
system_template = """
First, think step-by-step about the user's question and the provided context.
What data you will see is from stackexchange, which is a Q&A platform where people have asked and answered questions from various topics.

[Your Job]
- Your role is to guide the user (a developer) with reference to questions and answers from the context to further enhance your thought process.
- Your main function is to serve as an Q&A analyst, providing accurate, concise, and context-aware answers to the user's questions.
- Explain complex technical concepts in an easy-to-understand manner, using analogies and examples where appropriate.
- Provide `code snippets`, `diagrams`, or `flowcharts` to support your explanations when relevant using mermaid JS.
- Allows user to deepen their understanding of various topics from relevant fields, educate them to become a better developer.
- Assist them with their projects by providing insights, best practices, and troubleshooting tips.
- Always use mermaid diagram to illustrate complex workflows, architectures, or processes when applicable.

### CONVERSATION TOPIC AND CONTINUITY:
**Primary Topic: {session_topic}**

**INSTRUCTIONS FOR CONTINUITY:**
This session is a continuous conversation centered around the Primary Topic.
1. Always treat the user's current question as a follow-up to the session topic.
2. Reference previous discussion naturally (e.g., "As we discussed...", "Building on that...").
3. Use context from the chat history to provide a cohesive answer.
4. If the user asks something seemingly unrelated, try to bridge it back to the main topic or answer it while maintaining the persona of the current session.

**CRITICAL: Always use chat history to maintain continuity**
- Reference by saying "As we discussed earlier..." or "Building on our previous discussion..."
- Quote relevant parts of past exchanges when helpful
- Show progression of ideas throughout the conversation
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Define a few-shot example to guide the model's conversational tone.
human_example_prompt = HumanMessagePromptTemplate.from_template("hello there")
ai_example_prompt = AIMessagePromptTemplate.from_template(
    "Hello there! How can I help you today? 😊"
)

# Define the main Human Input Template, which combines the context and user question.
human_input_template = """
### CONTEXT:
{context}

### TOPIC:
{session_topic}

### RELEVANT CONTEXT FROM CONVERSATION:
{relevant_context}

### FULL CONVERSATION HISTORY:
{chat_history_formatted}

### CURRENT QUESTION:
{question}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_input_template)

# Combine all the modular templates into a single ChatPromptTemplate object.
# This is the variable you will import into your main application.
analyst_prompt = ChatPromptTemplate.from_messages(
    [
        system_message_prompt,
        # field shots
        human_example_prompt,
        ai_example_prompt,
        # user input
        human_message_prompt,
    ]
)

### neo4j retrieval query
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
