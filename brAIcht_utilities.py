import time

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader


def extract_data_and_create_retriever():
    """
    Constructs a retrieval system designed to store and access relevant 
    documents based on user input using FAISS vectorstore.
    
    Args:
        None

    Returns:
        retriever: A system that integrates hybrid search methods and 
        reranking techniques to enhance search efficiency.
    """
    
    file_paths = ["brecht_dataset/arturo.txt", "brecht_dataset\Baal.txt", "brecht_dataset\Badner Lehrstuck.txt",
                "brecht_dataset\corolian.txt", "brecht_dataset\Der kaukasische Kreidekreis.txt", 
                "brecht_dataset\Dialoge aus dem Messingkauf.txt",
                "brecht_dataset\die Ausnahme und die Regel.txt", "brecht_dataset\Die Bibel.txt", 
                "brecht_dataset\Die heilige Johanna der Schlachthofe .txt",
                "brecht_dataset\DIE RUNDKOPFE UND DIE SPITZKOPFE.txt", 
                "brecht_dataset\Die sieben Todsünden der Kleinbürger.txt",
                "brecht_dataset\Die-Gewehre der Frau Carrar.txt", "brecht_dataset\Don.txt", 
                "brecht_dataset\dreigroschenoper-2908-3.txt",
                "brecht_dataset\Guter Mensch Sezuan.txt", "brecht_dataset\Herr Puntila und sein Knecht Matti.txt",
                "brecht_dataset\hochzeit.txt", "brecht_dataset\Im Dickicht der Städte.txt", "brecht_dataset\jasager.txt",
                "brecht_dataset\Leben des Galilei.txt", "brecht_dataset\lindeberghs.txt", "brecht_dataset\Lukullus.txt",
                "brecht_dataset\mann-ist-mann-2.txt", "brecht_dataset\massnahme.txt", 
                "brecht_dataset\Mutter Courage und ihre Kinder.txt",
                "brecht_dataset\Mutter.txt", "brecht_dataset\Schweyk.txt", "brecht_dataset\Simone.txt", 
                "brecht_dataset\TrommelnDerNacht.txt",
                ]
        
    #file_path = "documents\Internship_Report.pdf"

    plays = []

    for path in file_paths:
        loader = TextLoader(path, encoding='utf-8')
        data = loader.load()
        plays.extend(data)  

    #plays_text = [doc.page_content for doc in plays]
    #combined_text = " ".join(plays_text)

    #loader = PyPDFLoader(file_path)
    #data = loader.load()
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #semantic_chunker = SemanticChunker(
    #    embeddings,
    #    breakpoint_threshold_type="percentile",
    #)
    #docs = semantic_chunker.create_documents([combined_text])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    docs = text_splitter.split_documents(plays)
    
    #vectorstore = FAISS.from_documents(plays, embeddings)
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
            
    similarity_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 5
    ensemble_retriever = EnsembleRetriever(retrievers=[similarity_retriever, keyword_retriever], 
                                        weights=[0.5, 0.5],
                                        )

    compressor = CohereRerank(model="rerank-multilingual-v3.0")

    retriever = ContextualCompressionRetriever(
        
        base_compressor=compressor, 
        base_retriever=ensemble_retriever,
    )
    
    return retriever


# A function to stream the output
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.06)
