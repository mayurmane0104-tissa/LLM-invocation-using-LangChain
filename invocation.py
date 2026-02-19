# # Simple LLM invocation and PromptTemplate example using LangChain

# import os
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI

# # Step 1: Initialize LLM
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# # Step 2: Create PromptTemplate
# template = """
# You are a helpful assistant. Answer the question about {topic} in {style}.
# Question: {question}
# """
# prompt = PromptTemplate.from_template(template)

# # Step 3: Format and invoke
# formatted_prompt = prompt.format(topic="tokens", style="simple terms", question="What is a token?")
# response = llm.invoke(formatted_prompt)

# print(response.content)

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2")  # Install ollama.com + `ollama pull llama3.2`
template = """
Answer about {topic}: {question}
"""
prompt = PromptTemplate.from_template(template)
response = llm.invoke(prompt.format(topic="tokens in llm", question="What is it?"))
print(response)
