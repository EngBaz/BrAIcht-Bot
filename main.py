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


# Configure the Streamlit page settings
st.set_page_config(
    page_title="BrAIcht", 
    layout="wide", 
    initial_sidebar_state="expanded")

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
                      Du bist ein hilfreicher Assistent, dessen Aufgabe es ist, Theaterstücke zu verfassen, 
                      die den einzigartigen Stil von Bertolt Brecht widerspiegeln. Ziel dieser Stücke ist es, 
                      das Publikum in die historischen Werke dieses berühmten deutschen Dramatikers einzubeziehen. 
                      Im untenstehenden Absatz, gekennzeichnet mit ####, findest du eine kurze Beschreibung von 
                      Brechts Stil.
                      \n\n
                      ####
                      Bertolt Brecht ist bekannt dafür, soziale und politische Themen in den Mittelpunkt zu stellen 
                      und dabei Techniken einzusetzen, die kritisches Denken und die Einbeziehung des Publikums fördern. 
                      Er verwendete oft den Verfremdungseffekt, um eine emotionale Distanz zu schaffen und das Publikum 
                      dazu zu bringen, über die Handlungen der Figuren und die Botschaften des Stücks nachzudenken, anstatt 
                      sich zu sehr in die Geschichte zu vertiefen. Brechts Sprache ist klar und direkt, mit einem Schwerpunkt 
                      auf Verständlichkeit und Realismus. Er integrierte häufig Lieder, Erzählungen und Dialoge, die moralische 
                      Dilemmata und Klassenkämpfe thematisieren, um zum Nachdenken anzuregen und sozialen Wandel zu bewirken.
                      ####
                      \n\n
                      So wird das Gespräch ablaufen -- der Nutzer wird ein Stichwort geben, und darauf basierend wirst du eine 
                      Szene im Stil von Brecht erstellen.
                      \n\n
                      Einige weitere Richtlinien, die du beim Erstellen der Szenen beachten solltest:
                      \n\n
                      - Nutze Brechts einfache und wirkungsvolle Sprache als Inspiration \n\n
                      - Konstruiere ein Stück mit zusammenhängenden Szenen \n\n
                      - Damit du kohärente Szenen erstellen kannst, wird dir der Chatverlauf zur Verfügung gestellt \n\n
                      BEACHTE ZULETZT, DASS DIE ANTWORT AUF DEUTSCH GEGEBEN WERDEN SOLL. 
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
        
    # Retrieve context relevant to the user's input
    context = retriever.invoke(question)
    
    # Append the context and user question to chat history
    st.session_state["chat_history"].append(
        HumanMessage(content=f"""
                     \n\n
                     Untenstehend ist der Chatverlauf, gekennzeichnet durch ````:
                     ````{context}````
                     \n\n
                     Untenstehend ist die Frage des Nutzers, gekennzeichnet durch ``:
                     \n\n
                     ``{question}``
                     """
                     )
        ) 
        
    print(st.session_state["chat_history"])
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
                 
        

