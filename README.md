# BrAIcht Bot with LangGraph, FAISS, Groq Llama 3 and Streamlit

This project presents BrAIcht, a chatbot which objective is to create plays in the style of the famous German playwright Bertolt Brecht.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [References](#References)

## Introduction

The BrAIcht project introduces an AI chatbot that creates plays in the style of the famous German playwright Bertolt Brecht. 
The project utilizes Retrieval Augmented Generation (RAG) techniques using GROQ Llama 3.1, LangChain and Streamlit.

## Setup

To setup this project on your local machine, follow the below steps:
1. Clone this repository: <code>git clone github.com/EngBaz/Hybrid-RAG-System</code>

2. Create a virtual enviromnent
   ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate.bat
    ```
3. Install the required dependencies by running <code>pip install -r requirements.txt</code>

4. Obtain an API key from OpenAI, Cohere AI and Groq. Store the APIs in a <code>.env</code> file as follows:
    ```console
    
    $ GOOGLE_API_KEY="your api key"
    $ GROQ_API_KEY="your api key"
    $ COHERE_API_KEY="your api key"
    ```

## Usage

To use the conversational agent:
1. In the terminal, run the streamlit app: <code> streamlit run main.py </code>
2. Select the appropriate format of your file 
3. Upload your file
4. Write a specific question about the uploaded file
5. The agent will process the input and respond with relevant information
6. 

## References

[1] Hybrid RAG: https://arxiv.org/pdf/2408.05141

[2] Rerankers: https://arxiv.org/abs/2409.07691

[3] Corrective-RAG: https://arxiv.org/abs/2401.15884

[4] https://python.langchain.com/docs/tutorials/rag/

[5] https://github.com/mistralai/cookbook/tree/main/third_party/langchain



