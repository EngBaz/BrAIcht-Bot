import time
import streamlit as st
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

@st.cache_resource
def extract_data_and_create_retriever(size, overlap):
    
    """ Constructs a retrieval system designed to store and access relevant 
    documents based on user input using FAISS vectorstore. """
    
    file_paths = ["data/brecht_plays/arturo.txt", "data/brecht_plays/Baal.txt", "data/brecht_plays/Badner Lehrstuck.txt",
                "data/brecht_plays/corolian.txt", "data/brecht_plays/Der kaukasische Kreidekreis.txt", 
                "data/brecht_plays/Dialoge aus dem Messingkauf.txt",
                "data/brecht_plays/die Ausnahme und die Regel.txt", "data/brecht_plays/Die Bibel.txt", 
                "data/brecht_plays/Die heilige Johanna der Schlachthofe .txt",
                "data/brecht_plays/DIE RUNDKOPFE UND DIE SPITZKOPFE.txt", 
                "data/brecht_plays/Die sieben Todsünden der Kleinbürger.txt",
                "data/brecht_plays/Die-Gewehre der Frau Carrar.txt", "data/brecht_plays/Don.txt", 
                "data/brecht_plays/dreigroschenoper-2908-3.txt",
                "data/brecht_plays/Guter Mensch Sezuan.txt", "data/brecht_plays/Herr Puntila und sein Knecht Matti.txt",
                "data/brecht_plays/hochzeit.txt", "data/brecht_plays/Im Dickicht der Städte.txt", "data/brecht_plays/jasager.txt",
                "data/brecht_plays/Leben des Galilei.txt", "data/brecht_plays/lindeberghs.txt", "data/brecht_plays/Lukullus.txt",
                "data/brecht_plays/mann-ist-mann-2.txt", "data/brecht_plays/massnahme.txt", 
                "data/brecht_plays/Mutter Courage und ihre Kinder.txt",
                "data/brecht_plays/Mutter.txt", "data/brecht_plays/Schweyk.txt", "data/brecht_plays/Simone.txt", 
                "data/brecht_plays\TrommelnDerNacht.txt",
                ]

    plays = []

    for path in file_paths:
        loader = TextLoader(path, encoding='utf-8')
        data = loader.load()
        plays.extend(data)
        
    play_texts = [doc.page_content for doc in plays]
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    docs = semantic_chunker.create_documents(play_texts)
    
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    #docs = text_splitter.split_documents(plays)
    
    vectorstore = FAISS.from_documents(docs, embeddings)
            
    similarity_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 5
    ensemble_retriever = EnsembleRetriever(retrievers=[similarity_retriever, keyword_retriever], 
                                        weights=[0.5, 0.5])

    compressor = CohereRerank(model="rerank-multilingual-v3.0")

    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
    
    return retriever

# A function to stream the output
def stream_data(response):
    
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.06)
