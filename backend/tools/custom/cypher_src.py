### neo4j retrieval query
retrieval_query = """
// Start from vector search result variables: `node`, `score`
WITH node, score

// 1. Efficient Routing using Label Filters
// By checking `WHERE node:Label` first, Neo4j safely skips executing non-matching branches entirely.
CALL {
  WITH node
  WITH node WHERE node:Question
  RETURN node AS q
  UNION
  WITH node
  WITH node WHERE node:Answer
  MATCH (node)-[:ANSWERS]->(q:Question)
  RETURN q
  UNION
  WITH node
  WITH node WHERE node:Tag
  MATCH (q:Question)-[:TAGGED]->(node)
  RETURN q
  UNION
  WITH node
  WITH node WHERE node:User
  MATCH (node)-[:ASKED]->(q:Question)
  RETURN q
  UNION
  WITH node
  WITH node WHERE node:User
  MATCH (node)-[:PROVIDED]->(:Answer)-[:ANSWERS]->(q:Question)
  RETURN q
}
WITH DISTINCT q AS question, node, score
WHERE question IS NOT NULL

// 2. Community detection overlap filter
WITH question, node, score,
  coalesce(question.CommunityId, []) AS qComm,
  coalesce(node.CommunityId, []) AS nComm
WITH question, node, score, qComm, nComm,
  any(x IN qComm WHERE x IN nComm) AS sameCommunity,
  (size(qComm) > 0 AND size(nComm) > 0) AS bothHaveCommunity
WHERE NOT bothHaveCommunity OR sameCommunity

// 3. Pattern Comprehension for lightning-fast sub-graph extraction
// This eliminates Cartesian Products and expensive OPTIONAL MATCH / COLLECT chains.
WITH question, score, sameCommunity, qComm, nComm,
  {
    id: question.id,
    title: question.title,
    body: question.body,
    link: question.link,
    score: question.score,
    favorite_count: question.favorite_count,
    creation_date: toString(question.creation_date)
  } AS questionDetails,
  
  // Extract Askers instantly without expanding row counts
  head([(asker:User)-[:ASKED]->(question) | {
    id: asker.id,
    display_name: asker.display_name,
    reputation: asker.reputation
  }]) AS askerDetails,
  
  // Extract Tags
  [(question)-[:TAGGED]->(tag:Tag) | tag.name] AS tags,
  
  // Extract Answers & their Providers as nested comprehensions
  [(answer:Answer)-[:ANSWERS]->(question) | {
    id: answer.id,
    body: answer.body,
    score: answer.score,
    is_accepted: answer.is_accepted,
    creation_date: toString(answer.creation_date),
    provided_by: head([(provider:User)-[:PROVIDED]->(answer) | {
      id: provider.id,
      display_name: provider.display_name,
      reputation: provider.reputation
    }])
  }] AS answers

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
