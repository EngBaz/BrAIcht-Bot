## BrAIcht, a theatrical agent that speaks like Bertolt Brecht

The BrAIcht project introduces an AI chatbot that creates plays in the style of the famous German playwright Bertolt Brecht. 
The project utilizes Retrieval Augmented Generation (RAG) techniques using <code>LLaMA 3</code>, <code>FAISS</code>, <code>LangChain</code>, and <code>Streamlit</code>.

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

This project implements <code>QLoRA</code> to fine-tune large language models with two datasets: one with over <code>540 000</code> data points from German plays and another with more than <code>17 000</code> data points from Brecht's works. The performance of the model is then evaluated using <code>BLEU score</code> and <code>perplexity</code>.

Additionally, RAG pipeline is developped to reduce hallucinations and to extract context from a <code>FAISS</code> vectorDB containing plays by Bertolt Brecht. This context, together with the chat history, will be refined through prompt engineering and few-shot prompting to help the language model generate text in Brecht's style.

The text data is prepared for indexing with the <code>semantic chunkung</code>, which splits the documents into coherent chunks for effective retrieval. Then, a hybrid search strategy is used which combines <code>semantic search</code> with <code>cosine similarity</code> and <code>keyword search</code> with <code>BM25</code>. This dual approach improves retrieval accuracy by combining contextual understanding with keyword matching and providing relevant information to the LLM.

![RAG](images/rag_application.png)

## Related work

An article is published from this project in the GENERATIVE ART conference in Venice.
You can read the related paper [BrAIcht, a theatrical agent that speaks like Bertolt Brecht's characters](related_paper/BrAIcht.pdf).

Please not that the project in this github repo is an updated version of the published paper.

## References

[1] QLoRA: https://arxiv.org/abs/2305.14314

[2] Hybrid RAG: https://arxiv.org/pdf/2408.05141

[3] Rerankers: https://arxiv.org/abs/2409.07691

[4] Corrective-RAG: https://arxiv.org/abs/2401.15884

[5] LangChain: https://python.langchain.com/docs/tutorials/rag/




