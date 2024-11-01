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
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=temperature, max_tokens=max_new_tokens)
    
st.title("BrAIcht Bot🤖")

# Initialize a session state for storing chat history if it doesn't already exist
if "chat_history" not in st.session_state:
    
    st.session_state.chat_history = [
        
        SystemMessage(content= """
                      Du bist ein Assistent, dessen Aufgabe es ist, einen Dialog mit dem Benutzer im Stil des berühmten 
                      deutschen Dramatikers Bertolt Brecht zu führen. Dein Ziel ist es, das Publikum mit Szenen zu fesseln, 
                      die Brechts einzigartigen Stil widerspiegeln, indem du auf jede Eingabe des Benutzers mit einer einzigen, 
                      in sich geschlossenen Szene antwortest.
                      
                      Eine kurze Beschreibung von Brechts Stil findest du im folgenden Absatz, der von #### begrenzt wird:

                      ####
                      Bertolt Brecht konzentriert sich in seinen Werken auf soziale und politische Themen und nutzt den 
                      Verfremdungseffekt, um zum kritischen Denken anzuregen, anstatt emotionale Bindung zu schaffen. Seine 
                      Sprache ist direkt und realistisch und soll leicht verständlich sein. Brecht integriert oft Lieder, 
                      Erzählungen und Dialoge, die moralische Fragen und Klassenkonflikte behandeln und zur Reflexion und 
                      zum sozialen Wandel anregen.
                      #### 
                      
                      Um dir dabei zu helfen, findest du im Folgenden den Chatverlauf, der von '''' begrenzt wird. Jede Szene, 
                      die du erstellst, sollte nahtlos an die vorherigen anschließen, sodass am Ende ein zusammenhängendes und 
                      wirkungsvolles Stück entsteht. Stütze dich auf den Chatverlauf, um die Kontinuität und Konsistenz des Dialogs 
                      zu gewährleisten, da dies entscheidend für den Aufbau eines einheitlichen und bedeutungsvollen Stücks ist. 
                      Der Chatverlauf ist von '' begrenzt.
                      
                      BEACHTE, DASS DIE ANTWORT AUF DEUTSCH GEGEBEN WERDEN SOLL.
                      
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
if cue := st.chat_input("Enter a cue..."):
        
    # Store the user's question in the session messages
    st.session_state.messages.append({"role": "user", "content": cue})
        
    # Retrieve context relevant to the user's input
    context = retriever.invoke(cue)
    
    # Append the context and user question to chat history
    st.session_state["chat_history"].append(
        HumanMessage(content=f"""
                     Unten ist der Chatverlauf, abgegrenzt durch ````:
                     
                     ````{context}````
                     
                    Unten ist die Frage des Benutzers, abgegrenzt durch ``:
                     
                     ``{cue}``
                     
                     """
                     )
        ) 
        
    print(st.session_state["chat_history"])
    # Display the user's question in the chat
    with st.chat_message("user"): 
                
        st.write_stream(stream_data(cue))
            
    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        
        # Get the response from the model
        response = response = llm.invoke(st.session_state["chat_history"])
            
        # Display the assistant's response
        st.write_stream(stream_data(response.content))
            
        # Update chat history and messages sessions with the assistant's response
        st.session_state["chat_history"].append(AIMessage(content=response.content))
        st.session_state["messages"].append({"role": "assistant", "content": response.content}) 
                 
        

