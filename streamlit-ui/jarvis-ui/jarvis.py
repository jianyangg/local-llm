import json
import os
import requests
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from app_config import config
from termcolor import cprint
from utils import generate_tenant_id, delete_screenshots, draw_bounding_box_on_pdf_image
from yaml.loader import SafeLoader

st.set_page_config(page_title="Jarvis", page_icon="🤖")

with open("style.css", "r") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


with open('auth/config.yaml') as file:
    auth_config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    auth_config['credentials'],
    auth_config['cookie']['name'],
    auth_config['cookie']['key'],
    auth_config['cookie']['expiry_days'],
    auth_config['pre-authorized']
)

# access the response from the orchestrator endpoint
def response_generator(prompt, tenant_id, chat_mode, generate_title=False, convo_hist=[]) -> json:
    print("Querying orchestrator...")
    url = config["orchestrator_url_entry"]
    # allow correct interpretation of data
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "tenant_id": tenant_id,
        "chat_mode": chat_mode,
        "generate_title": generate_title,
        "convo_hist": convo_hist
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_str = response.content.decode('utf-8')
    response_json = json.loads(response_str)
    cprint("Received response from orchestrator.", "green")

    # Return response and documents if available
    response = response_json.get('response')
    documents = response_json.get('documents')
    title = response_json.get('title')
    # remove aprostrophes
    if title:
        title = title.replace("'", "")
        title = title.replace('"', "")

    return response, documents, title

def fetch_image_paths():
    url = config["orchestrator_url_images"]
    response = requests.get(url)
    data = response.json()
    image_paths = data["images"]
    return image_paths

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
                yaml.dump(auth_config, file, default_flow_style=False)

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
    # Generate tenant_id based off username and password combination
    user_hashed_password = auth_config['credentials']['usernames'][username]['password']
    tenant_id = generate_tenant_id(username, user_hashed_password)
    past_convos = get_past_convos(tenant_id)

    authenticator.logout('Logout', 'main')

    st.write(f'Welcome *{name}*')

    st.title(f"Hello {name.split()[0]}, I am :rainbow[Jarvis].")

    # Initialize chat history for each user
    if "users_messages" not in st.session_state:
        st.session_state.users_messages = {}

    # Initialise previous convo title
    if "prev_convo_title" not in st.session_state:
        st.session_state.prev_convo_title = {}

    # Initialise conversation history if needed
    if username not in st.session_state.users_messages:
        st.session_state.users_messages[username] = []

    # Initiliase previous convo title
    if username not in st.session_state.prev_convo_title:
        st.session_state.prev_convo_title[username] = "Start"

    prev_convo_title = st.session_state.prev_convo_title[username]

    # Add chat mode selection to the sidebar
    with st.sidebar:
        chat_mode = st.selectbox("Select a chat mode", options=("Jarvis", "Standard RAG", "Standard Chatbot"), index=0)
        if chat_mode == "Jarvis":
            st.write("***Jarvis:** For document queries. High performance, slower speed.*")
        elif chat_mode == "Standard RAG":
            st.write("***Standard RAG:** For document queries. Medium performance, fast speed.*")
        elif chat_mode == "Standard Chatbot":
            st.write("***Standard Chatbot:** Just a regular chatbot powered by Meta Llama3.1.*")
        convo_title = st.selectbox("Select a conversation", options=past_convos, index=0 if prev_convo_title == "Start" else past_convos.index(prev_convo_title))
    # cprint(f"Previous convo title: {prev_convo_title}", "light_cyan")
    # cprint(f"Current convo title: {convo_title}", "light_cyan")

    # default value
    generate_title = False
    if prev_convo_title != convo_title:
        # chat changed
        # If change to New chat, reset chat history
        if convo_title == "New chat":
            st.session_state.users_messages[username] = []
            generate_title = True
        # If change to another chat name, load chat history
        else:
            st.session_state.users_messages[username] = load_convo(tenant_id, convo_title)
    else:
        # same chat
        # it could still be on the new chat or a different chat
        # but for both, we are still using the same chat history
        # which is still st.session_state.users_messages[username]
        # do nothing
        pass

    # Display chat messages from history on app rerun
    for message in st.session_state.users_messages[username]:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You</strong><p>{message["content"]}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>Jarvis ✧₊⁺</strong><p>{message["content"]}</p></div>', unsafe_allow_html=True)

    # Accept user input
    if prompt := st.chat_input(placeholder=f"Message Jarvis (mode: {chat_mode})", key="chat_input"):
        cprint("Clearing screenshots from previous session", "light_blue")
        # clear png files
        delete_screenshots(tenant_id)

        cprint("User message added to conversation history", "light_blue")

        # Display user message in chat message container
        st.markdown(f'<div class="user-message"><strong>You</strong><p>{prompt}</p></div>', unsafe_allow_html=True)

        with st.spinner("_Kicking into action..._"):
            response, docs_metadata, new_convo_title = response_generator(prompt=prompt, tenant_id=tenant_id, chat_mode=chat_mode, generate_title=generate_title, convo_hist=st.session_state.users_messages[username])
            # update convo title
            if generate_title:
                convo_title = new_convo_title
                cprint(f"Generated new convo title {convo_title}", "green")
            else:
                cprint(f"Using convo title {convo_title}", "green")
            
            # Update previous convo title
            prev_convo_title = convo_title
            # update session state
            st.session_state.prev_convo_title[username] = prev_convo_title

            response = "\n\n" + response.strip()
            # Display assistant response in chat message container
            st.markdown(f'<div class="assistant-message"><strong>Jarvis ✧₊⁺</strong><p>{response}</p></div>', unsafe_allow_html=True)

        # Only for certain chat modes that return documents
        if docs_metadata is not None:
            with st.sidebar:
                with st.spinner("_Processing source documents..._"):
                    # using docs_metadata, create a dictionary mapping documents to a list of sub-documents
                    # each sub-document is a dictionary with keys: page_content, file_path, page_idx, level, bbox
                    dict_of_docs = {doc["file_path"].split("/")[-1]: [] for doc in docs_metadata}
                    for doc in docs_metadata:
                        try:
                            image_path, caption, page_idx = draw_bounding_box_on_pdf_image(doc, location=f"output/{tenant_id}"), doc["file_path"].split("/")[-1], doc["page_idx"]
                            dict_of_docs[doc["file_path"].split("/")[-1]].append({"image_path": image_path, "caption": caption, "page_idx": page_idx})
                        except Exception as e:
                            # TODO: Bad code, using exception as part of the logic
                            print(e)
                            continue

                st.subheader(f"**Sources referenced ({len(dict_of_docs)} PDFs)**")
                idx = 1
                for doc in dict_of_docs:
                    if dict_of_docs[doc] == []:
                        continue
                    sub_docs = dict_of_docs[doc]
                    st.write(f"**{str(idx)}. {doc}**")
                    for sub_doc in sub_docs:
                        st.image(sub_doc["image_path"], caption=f"{sub_doc['caption']}, page {sub_doc['page_idx']}", use_column_width="auto")
                    idx += 1

        # Add user message to chat history
        st.session_state.users_messages[username].append({"role": "user", "content": prompt})
        # Add assistant response to chat history
        st.session_state.users_messages[username].append({"role": "assistant", "content": response})

        # Update the chat history in the json file
        cprint(f"Saving conversation history with title {convo_title}", "light_blue")
        if convo_title == "New chat":
            cprint("WRONG! It should not be New chat", "red")
        update_convo(tenant_id, st.session_state.users_messages[username], convo_title)
        # cprint(f"Conversation History is {st.session_state.users_messages[username]}", "light_blue")
        cprint("Conversation history saved.", "light_blue")


def get_past_convos(tenant_id):
    # returns a list of convo topics
    # Convo is short for conversation
    default = ["New chat"]
    # This function reads the json file for chat history
    convo_path = f"chat_history/{tenant_id}"
    if not os.path.exists(convo_path):
        # create dir
        os.makedirs(convo_path)
        return default
    
    files = os.listdir(convo_path)
    if files is not None:
        default.extend(file.split(".")[0] for file in files)

    return default

def load_convo(tenant_id, convo_title):
    # go to file and parse json convo history as list of dictionaries
    # assumes the file already exists
    convo_path = f"chat_history/{tenant_id}/{convo_title}.json"
    try:
        with open(convo_path) as file:
            convo = json.load(file)
            print("Loaded conversation:")	
            print(convo)
            print()
            return convo

    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return []
    
def update_convo(tenant_id, convo, convo_title):
    # if convo_title is new, create new json file
    # write convo to json file
    if '"' in convo_title:
        convo_title = convo_title.replace('"', "")
    convo_path = f'chat_history/{tenant_id}/{convo_title}.json'
    directory = os.path.dirname(convo_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(convo_path, "w") as file:
        json.dump(convo, file)
        
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