from langchain_core.prompts import ChatPromptTemplate

ROUTER_SYSTEM_PROMPT = """
You are an expert at routing functionality.
Use the following criteria to decide how to route the user's input:

1. **retrieval_needed**: The user is asking a specific technical question about programming, code, libraries, errors, or requesting information that would likely be found in a StackOverflow database (questions, answers, tags, users). Examples: "How do I reverse a list in Python?", "What is a NullPointerException?", "Who is the top user for python tag?".

2. **conversational**: The user is engaging in general conversation, greeting, asking about your identity, or asking questions that do NOT require looking up specific technical data from the knowledge base. Examples: "Hi", "How are you?", "What can you do?", "Write me a poem".

Return ONLY 'retrieval_needed' or 'conversational'. do not return anything else."""

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ROUTER_SYSTEM_PROMPT,
        ),
        ("human", "{question}"),
    ]
)
