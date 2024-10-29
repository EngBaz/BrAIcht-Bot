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
                      
                      You are a helpful assistant whose job it is to create a conversation between you and the user. 
                      Here's how it works: the user gives a cue and your response is a dialogue that reflects the unique 
                      style of Bertolt Brecht.
                      Your goal is to engage the audience in plays that are similar in style to the famous German playwright's 
                      plays. A brief description of Brecht's style can be found in the following paragraph marked ####: 
                      #### 
                      Bertolt Brecht is known for focusing on social and political issues and using techniques that encourage 
                      critical thinking and audience participation. He often uses the alienation effect to create emotional 
                      distance and get the audience to think about the characters' actions and the play's messages rather 
                      than getting too engrossed in the story. Brecht's language is clear and direct, with an emphasis on 
                      understandability and realism. He often incorporated songs, narratives and dialogues that address moral 
                      dilemmas and class struggles to provoke thought and bring about social change.
                      #### 
                      How the conversation works: The user enters a clue and based on that clue we provide you with a context 
                      that is retrieved from a vector database and is dialogues by Bertolt Brecht.
                      The context is provided to you below and is delimited by ' '. Make sure that every time you create a dialogue 
                      it is coherent with the prescribed scenes as in the end you should be able to provide the user with a very 
                      coherent and meaningful piece. For this purpose you are also provided with the chat history which you should 
                      also rely on to create this coherence in the dialogues. The chat history is delimited by ''.
                      
                      LAST BUT NOT LEAST NOTE THAT THE ANSWER SHOULD BE GIVEN IN GERMAN.
                      
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
                 
        

