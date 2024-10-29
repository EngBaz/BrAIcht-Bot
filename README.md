# BrAIcht, a theatrical agent that speaks like Bertolt Brecht

The BrAIcht project introduces an AI chatbot that creates plays in the style of the famous German playwright Bertolt Brecht. 
The project utilizes Retrieval Augmented Generation (RAG) techniques using Groq Llama 3.2, FAISS, LangChain, and Streamlit.

![brAIcht](images/brAIcht.png)

## Setup

To setup this project on your local machine, follow the below steps:
1. Clone this repository: <code>git clone github.com/EngBaz/BrAIcht-Bot</code>

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
5. run the streamlit app: <code> streamlit run main.py </code>

## Implementation

In this project, a <code>Retrieval Augmented Generation (RAG)</code> pipeline is developed that extracts the corresponding context in response to a cue from a vector database. The retrieved information includes the plays of <code>Bertolt Brecht</code>. This context is then forwarded to the language model (LLM) together with the chat history. In addition, the extracted context is enhanced by prompt engineering and "few-shot" prompting techniques so that the LLM is able to generate scenes and dialogues that are stylistically similar to Brecht's plays.

The vector store used in this project is <code>FAISS (Facebook AI Similarity Search)</code>, an efficient library for similarity search and clustering of dense vectors. FAISS is designed to handle large datasets and provides multiple algorithms for searching in high-dimensional spaces, making it particularly effective for machine learning and information retrieval applications. Its capabilities allow for fast similarity searches that enable the retrieval of relevant documents or information based on vector representations.

To prepare the text data for indexing, we use the <code>RecursiveCharacterTextSplitter</code> technique. In this approach, the documents are split into smaller, manageable chunks based on the number of characters, taking into account the natural structure of the text. By recursively splitting the content, the technique ensures that the chunks are in a coherent context, allowing for more effective retrieval and better understanding when processed by the LLM.

We implement a <code>hybrid search</code> strategy for searching and retrieving relevant information from the FAISS vector store. This method combines <code>semantic search</code>, which uses <code>cosine similarity</code> to evaluate the proximity of vector representations, with <code>keyword search</code> based on <code>BM25</code>. Semantic search identifies relevant chunks by measuring the angle between vectors in the embedding space, effectively capturing nuanced meanings and relationships. The BM25 algorithm enhances this process by ranking documents based on keyword matches and <code>TF-IDF (Term Frequency - Inverse Document Frequency)</code> metrics. Together, these approaches ensure that the retrieval process is both contextual and precise, resulting in more accurate and relevant information being presented to the LLM.

## Related work

We published an article issued from this project in the GENERATIVE ART conference in Venice.
You can read the related paper [BrAIcht, a theatrical agent that speaks like Bertolt Brecht's characters](related_paper/BrAIcht.pdf).

Please not that the project in this github repo is an updated version of the published paper.

## References

[1] QLoRA: https://arxiv.org/abs/2305.14314

[2] Hybrid RAG: https://arxiv.org/pdf/2408.05141

[3] Rerankers: https://arxiv.org/abs/2409.07691

[4] Corrective-RAG: https://arxiv.org/abs/2401.15884

[5] https://python.langchain.com/docs/tutorials/rag/




