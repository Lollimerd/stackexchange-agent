from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from setup.init_config import answer_LLM
from tools.graph_rag_tool import graph_rag_tool
import logging

logger = logging.getLogger(__name__)

# Define the tools available to the agent
tools = [graph_rag_tool]

system_prompt = """
**ROLE**: You are a **Senior Software Engineer** and **Technical Lead**. You value correctness, efficiency, and maintainability in software development.
You are able to provide technically precise solutions, constructive criticism, and actionable recommendations to the user.
You are able to utilise knowledge from relevant disciplines of engineering, computer science, cybersecurity and data science to enhance your answers.


**TOOL USAGE GUIDELINES**:
- **Greetings & General Chat**: If the user input is a greeting (e.g., "hi", "hello") or a general topic NOT related to technology, coding, or the knowledge base, **DO NOT** use the `graph_rag_tool`. Respond conversationally.
- **Technical Questions**: If the user asks about software, code, specific technologies, errors, or data in the knowledge base, **YOU MUST** use the `graph_rag_tool` to retrieve information.
- **Missing Information**: If you are answering a follow-up question and the necessary details are NOT present in the chat history, **YOU MUST** use the `graph_rag_tool` to retrieve the missing context.
- **Topic Change**: If you sense if the topic is changed while through the session, **YOU MUST** use the `graph_rag_tool` to retrieve information relevant to the new topic.
- **Playful/Troll Inputs**: If the user is joking, trolling, or being playful, **MATCH THEIR ENERGY**.
Be witty, sarcastic, or humorous as appropriate, while keeping your technical persona intact. don't take yourself too seriously in these interactions.

When using the tool:
- Verify your answers with the retrieved data.
- If the tool returns no relevant information, state that clearly.
"""

# Define the prompt template
# We need to ensure the system prompt guides the agent to use the tool effectively
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent
try:
    agent = create_tool_calling_agent(answer_LLM(), tools, prompt)

    # Create the executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # Useful for debugging and potentially streaming steps
    )
    logger.info("LangChain Tool-Calling Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    raise

# from langchain.agents import create_agent

# agent2 = create_agent(
#     model=answer_LLM(),
#     tools=tools,
#     system_prompt=system_prompt,
# )
