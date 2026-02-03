from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from setup.init_config import answer_LLM
from tools.graph_rag_tool import graph_rag_tool
import logging

logger = logging.getLogger(__name__)

# Define the tools available to the agent
tools = [graph_rag_tool]

# Define the prompt template
# We need to ensure the system prompt guides the agent to use the tool effectively
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant assisting with StackExchange data analysis. "
            "Use the graph_rag_tool to retrieve information from the knowledge base. "
            "Always verify your answers with the tool data. "
            "If you cannot find the answer in the tool output, say so.",
        ),
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
