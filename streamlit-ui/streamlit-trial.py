import streamlit as st
import random
import time
from langchain_community.chat_models import ChatOllama
from rag_workflow import stuff

config={"ollama_base_url": "http://localhost:11434",
        "llm_name": "llama3",
        "neo4j_url": "neo4j://localhost:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "password",
        "pdf_path": "data/mcbook-user-guide.pdf",		
        }

# load the llm
llm = ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=config["llm_name"],
            streaming=True,
            # seed=2,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )

# Streamed response emulator
def response_generator(prompt):
    stuff(st)
    # TODO: implement custom workflow here.
    # yield the print statements when deciding on the response
    # create another clasas in another file, pass the instance there.
    response = llm.invoke(input=prompt).content
    for word in response.split():
        yield word + " "
        time.sleep(0.025)


st.title("Basic Chatbot with Streamlit")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter a prompt here."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt=prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})