import os
import streamlit as st

from rag_utilities import *
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

# Set API keys for the services used (GROQ, Cohere, Google and OpenAI) from environment variables 
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
 
if __name__=="main":

    # Configure the Streamlit page settings
    st.set_page_config(
        page_title="BrAIcht", 
        layout="wide", 
        initial_sidebar_state="expanded",
        )

    # Create a sidebar for model settings
    with st.sidebar:
            
        st.header("Model Settings") 
        max_new_tokens = st.number_input("Select a max token value:", min_value=1, max_value=8192, value=1000)
        temperature = st.number_input("Select a temperature value:", min_value=0.0, max_value=2.0, value=0.0)
        
    # Load the LLM from Groq with specified parameters
    llm = ChatGroq(model="llama-3.2-3b-preview", temperature=temperature, max_tokens=max_new_tokens)
    
    st.title("BrAIcht Bot🤖")

    # Initialize a session state for storing chat history if it doesn't already exist
    if "chat_history" not in st.session_state:
        
        st.session_state.chat_history = [
            
            SystemMessage(content= """
                          
        Du bist ein hilfreicher Assistent, der den Stil des berühmten Dramatikers Bertolt Brecht imitiert. 
        Deine Aufgabe ist es, Dialoge zu erstellen, die Brechts einzigartigen Stil widerspiegeln, der sich 
        oft mit politischen, sozialen und philosophischen Themen befasst.
                          
        Du solltest Klarheit, emotionale Distanz (Verfremdungseffekt) und ein realistisches Gefühl priorisieren, 
        um das kritische Denken des Publikums anzuregen.

        Beim Erstellen der Dialoge:
        -Verwende Brechts einfache, aber effektive Sprache als Leitfaden.
        -Beziehe Themen wie Gesellschaftskritik, Klassenkampf oder moralische Dilemmata ein.
        -Erstelle Dialoge, die in eine Szene eines Stücks passen, und verwende kurze, zielgerichtete Sätze.
        -Halte die Antwort kurz und beschränke sie auf maximal drei Sätze. Verwende den folgenden Chatverlauf als Leitfaden:

        Die Antwort muss auf Deutsch sein.
                          
        """
        )
    ]

    if "messages" not in st.session_state:
        
        st.session_state.messages = []
    
    # Display previous messages in the chat UI 
    for message in st.session_state.messages:
        
        with st.chat_message(message["role"]):
            st.markdown(message["content"])  
        
    retriever = extract_data_and_create_retriever()

    # Capture user input for questions
    if question := st.chat_input("Enter a message"):
        
        # Store the user's question in the session messages
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Retrieve context relevant to the user's question
        context = retriever.invoke(question)
        
        # Append the context and user question to chat history
        st.session_state["chat_history"].append(
            HumanMessage(content=f"\n\nNachfolgend der Kontext:\n\n{context}\n\n Nachfolgend die Frage des Benutzers:\n\n{question}")
        ) 
        
        # Display the user's question in the chat
        with st.chat_message("user"): 
                
            st.write_stream(stream_data(question))
            
        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            
            # Get the response from the model
            response = response = llm.invoke(st.session_state["chat_history"])
            
            # Display the assistant's response
            st.write_stream(stream_data(response.content))
            
            # Update chat history and messages sessions with the assistant's response
            st.session_state["chat_history"].append(AIMessage(content=response.content))
            st.session_state["messages"].append({"role": "assistant", "content": response.content}) 
              
        
        

