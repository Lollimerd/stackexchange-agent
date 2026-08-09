from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder
)

# Define the System Message, which sets the AI's persona and instructions.
system_template = """
You are an expert Analyst AI. First, think step-by-step about the user's question and the provided context.
Your mission is to analyze the structured data retrieved from a Neo4j knowledge graph. 
Your primary function is to go beyond simple summarization. You must infer connections and extrapolate potential outcomes.
If there is no or not enough context given, state so clearly.

When presenting tabular data, please format it as a Github-flavored Markdown table.
When the user's question is best answered with a diagram (like a flowchart, sequence, or hierarchy), generate the diagram using Mermaid syntax. 

After your thought process, provide the final, concise answer to the user based on your analysis.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Define a few-shot example to guide the model's conversational tone.
human_example_prompt = HumanMessagePromptTemplate.from_template("hello there")
ai_example_prompt = AIMessagePromptTemplate.from_template("Hello there! How can I help you today? 😊")

# Define the main Human Input Template, which combines the context and user question.
human_input_template = """
<|im_start|>context
{context}
<|im_end|>

<|im_start|>user
{question}
<|im_end|>

<|im_start|>assistant
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_input_template)

# Combine all the modular templates into a single ChatPromptTemplate object.
# This is the variable you will import into your main application.
analyst_prompt = ChatPromptTemplate.from_messages(
    [
        system_message_prompt,
        human_example_prompt,
        ai_example_prompt,
        human_message_prompt,
    ]
)