import streamlit as st
import os
import database_api

st.set_page_config(page_title="file_uploader", page_icon="📂")

# Prevents users from uploading before signing in.
if not st.session_state.authentication_status:
    st.info('Please Login from the Home page and try again.')
    st.stop()

st.title("Upload a file")
st.write("Upload a file to the database.")

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

# accept multiple files
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    # st.write(bytes_data)

    # create folder documents if it doesn't exist
    if not os.path.exists("documents"):
        os.makedirs("documents")

    # save the file in a folder documents if it doesn't exist
    with open(f"documents/{uploaded_file.name}", "wb") as f:
        f.write(bytes_data)
        st.write("File received.")

if len(uploaded_files) != 0:
    columns = st.columns([3, 1])
    with columns[1]:
        if st.button("Upload", use_container_width=True):
            try:
                with st.spinner("Uploading..."):
                    # TODO: Make this asynchronous
                    st.write("Do not send queries while database is being updated.")
                    # TODO: Tag the tenant id to the uploaded files
                    # TODO: Access this via (st.session_state.username)
                    print(f"Tenant ID: {st.session_state['username']}")
                    database_api.upload_files(uploaded_files, tenant_id=st.session_state["username"])
                st.success("Files uploaded successfully.")
            except Exception as e:
                st.error(f"Error: {e}")

        
