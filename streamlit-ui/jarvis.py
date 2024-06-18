import streamlit as st
# from streamlit_chat import message
import time
from entryPoint import entry
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="Jarvis", page_icon="🤖")

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

# Streamed response emulator
def response_generator(prompt, tenant_id):
    response = entry(prompt, st, tenant_id)
    print("TEXT:", response)
    lines = response.split("\n")

    # Animation
    for line in lines:
        yield line + "\n"
        time.sleep(0.042)


name, authentication_status, username = authenticator.login('main', fields = {'Form name': 'Welcome to Jarvis.'})

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')

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
            with st.spinner("Thinking..."):
                response = st.write_stream(response_generator(prompt=prompt, tenant_id=username))

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

