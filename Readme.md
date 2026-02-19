LangChain LLM Invocation – Local & OpenAI Integration

1. Project Overview

This repository demonstrates a structured implementation of LLM invocation using LangChain.

The objective of this exercise was to:

    - Understand LLM invocation flow

    - Implement PromptTemplate usage

    - Compare cloud-based vs local model execution

    - Deliver a working prototype

Two approaches were considered:

    - OpenAI API (Cloud-based LLM)

    - Ollama (Local LLM runtime)

Due to OpenAI API billing restrictions on the current account, the final working implementation uses a locally hosted LLM via Ollama.

2. Architecture Overview

    User Input Variables
            ↓
    PromptTemplate (LangChain)
            ↓
    Formatted Prompt
            ↓
    LLM (OpenAI or Ollama)
            ↓
    Model Response
            ↓
    Terminal Output


LangChain acts as the orchestration layer between prompt engineering and model invocation.

3. Implementation Details

    3.1 PromptTemplate Usage

    A reusable prompt template was created:

        template = """
        Answer about {topic}: {question}
        """


        Dynamic variables:

        {topic}

        {question}

    This ensures separation between prompt structure and input data.

4. Local LLM Implementation (Working Version)

    Technology Stack-

        - Python

        - LangChain

        - Ollama

        - Llama 3.2

Code

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import OllamaLLM

    llm = OllamaLLM(model="llama3.2")

    template = """
    Answer about {topic}: {question}
    """

    prompt = PromptTemplate.from_template(template)

    response = llm.invoke(
        prompt.format(
            topic="tokens in llm",
            question="What is it?"
        )
    )

    print(response)


- Execution Flow

    - Initialize local LLM

    - Create structured prompt

    - Format prompt with runtime values

    - Send formatted prompt to model

    - Receive and print output

5. OpenAI Cloud-Based Version (Initial Approach)

Intended Code (OpenAI Version)

    import os
    from dotenv import load_dotenv
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    load_dotenv()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    template = """
    You are a helpful assistant.
    Answer about {topic}: {question}
    """

    prompt = PromptTemplate.from_template(template)

    response = llm.invoke(
        prompt.format(
            topic="tokens in llm",
            question="What is it?"
        )
    )

    print(response.content)


Why It Was Not Used

    -  The OpenAI version required:

        - Active billing setup

        - Valid API key with payment method attached

        - During execution, API calls failed due to billing restrictions on the account.

    - As a result:

        - No cloud-based invocation was possible

        - To complete the deliverable without delay, a local LLM approach was adopted

6. Rationale for Using Ollama (Local LLM)

    - To ensure:

        - No dependency on paid APIs

        - No cloud billing limitations

        - Fully offline execution

        - Immediate working demonstration

        - Ollama was installed locally and llama3.2 was pulled:

        - ollama pull llama3.2


    - This allowed:

        - Successful LLM invocation

        - Completion of deliverable

        - Demonstration of prompt orchestration via LangChain

7. Key Learnings

    - LLM invocation pipeline in LangChain

    - Prompt engineering fundamentals

    - Difference between cloud and local LLM execution

    - Handling API billing limitations

    - Model abstraction layer via LangChain

8. How to Run (Local Version)

    - Install Ollama

    - https://ollama.com/

    - Pull Model
        ollama pull llama3.2

    - Install Dependencies
        pip install -r requirements.txt

    - Run Script
        python invocation.py

9. Future Improvements

    - Add streaming responses

    - Add structured output parsing

    - Integrate embeddings

    - Deploy as FastAPI service

    - Compare performance: OpenAI vs Ollama