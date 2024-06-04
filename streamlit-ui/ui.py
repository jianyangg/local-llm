import streamlit as st
# from streamlit_chat import message
import time
from entryPoint import entry

# Streamed response emulator
def response_generator(prompt):
    response = entry(prompt, st)
    print("TEXT:", response)
    lines = response.split("\n")

    # Animation
    for line in lines:
        yield line + "\n"
        time.sleep(0.042)

st.title("Hello, I am Jarvis.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
    # with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter a prompt here."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user", avatar="👤"):
        st.markdown("**You**")
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="🤖"):
        st.write("**Jarvis**")
        response = st.write_stream(response_generator(prompt=prompt), )

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

