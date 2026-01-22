from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from setup.init import ANSWER_LLM
from prompts.router_prompts import router_prompt

router_chain = (
    RunnablePassthrough.assign(question=lambda x: x["question"])
    | router_prompt
    | ANSWER_LLM
    | StrOutputParser()
)
