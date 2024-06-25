import streamlit as st
import requests
import json
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from utils import generate_tenant_id
from streamlit_authenticator.utilities.hasher import Hasher

st.set_page_config(page_title="Jarvis", page_icon="🤖")

# Custom CSS for better styling
st.markdown("""
    <style>
    .stChatInput {
        margin-bottom: 3px;
    }
    .stChatInput input {
        font-size: 16px;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #ccc;
        width: 10%;
    }
    .stChatInput input::placeholder {
        color: #888;
    }
    .stButton button {
        background-color: #003153;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .user-message {
        text-align: left;
        background-color: #003153;
        color: white;
        padding: 10px 10px 10px 15px;
        border-radius: 15px;
        margin: 10px 3px 10px auto;
        max-width: 50%;
    }
    .assistant-message {
        text-align: left;
        background-color: #f1f0f0;
        padding: 10px 10px 10px 15px;
        border-radius: 15px;
        margin: 10px auto 10px 3px;
        max-width: 70%;
    }
    </style>
    """, unsafe_allow_html=True)

with open('auth/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

# access the response from the orchestrator endpoint
def response_generator(prompt, tenant_id, chat_mode):
    url = "http://orchestrator:5001/entry"
    # alow correct interpretation of data
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "tenant_id": tenant_id,
        "chat_mode": chat_mode
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_text = response.content.decode('utf-8')

    return response_text

def login():
    name, authentication_status, username = authenticator.login('main', fields = {'Form name': 'Welcome to Jarvis.'})

    if authentication_status:
        home(username, name)

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

def register():
    # entered fields are temporarily stored in cookies
    try:
        email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user('main', pre_authorization=False)
        if email_of_registered_user:
            with open('auth/config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

            st.success('User registered successfully. Login to continue.')

    except Exception as e:
        st.error(e)

def auth(selection):
    if selection == "Login":
        login()
    elif selection == "Register":
        register()
    else:
        st.error('Please select an option')

def home(username, name):
    # Add chat mode selection to the sidebar
    with st.sidebar:
        chat_mode = st.radio("Select chat mode:", ("Jarvis", "Semantic Search w Agents", "Semantic Search w/o Agents", "Chatbot"))
    
    authenticator.logout('Logout', 'main')

    st.write(f'Welcome *{name}*')

    st.title(f"Hello {name.split()[0]}, I am Jarvis.")

    # Initialize chat history for each user
    if "users_messages" not in st.session_state:
        st.session_state.users_messages = {}

    # If the current user is not in the dictionary, add them
    if username not in st.session_state.users_messages:
        st.session_state.users_messages[username] = []

    # Display chat messages from history on app rerun
    for message in st.session_state.users_messages[username]:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You</strong><p>{message["content"]}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>Jarvis</strong><p>{message["content"]}</p></div>', unsafe_allow_html=True)

    # Accept user input
    if prompt := st.chat_input(placeholder=f"Message Jarvis (mode: {chat_mode})", key="chat_input"):
        # Add user message to chat history
        st.session_state.users_messages[username].append({"role": "user", "content": prompt})

        # Display user message in chat message container
        st.markdown(f'<div class="user-message"><strong>You</strong><p>{prompt}</p></div>', unsafe_allow_html=True)

        with st.spinner("_Kicking into action..._"):
            # Generate tenant_id based off username and password combination
            user_hashed_password = config['credentials']['usernames'][username]['password']
            tenant_id = generate_tenant_id(username, user_hashed_password)
            response = response_generator(prompt=prompt, tenant_id=tenant_id, chat_mode=chat_mode)
            # Display assistant response in chat message container
            st.markdown(f'<div class="assistant-message"><strong>Jarvis</strong><p>{response}</p></div>', unsafe_allow_html=True)

        # Add assistant response to chat history
        st.session_state.users_messages[username].append({"role": "assistant", "content": response})

if not st.session_state['authentication_status']:
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        auth("Login")
    with tab2:
        auth("Register")
else:
    username = st.session_state["username"]
    name = st.session_state["name"]
    home(username, name)