import database_api
import os
import requests
import streamlit as st
import yaml
from utils import generate_tenant_id
from yaml.loader import SafeLoader
from app_config import config
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer


try:

    st.set_page_config(page_title="Topics", page_icon="💡")

    # Prevents users from uploading before signing in.
    if not st.session_state['authentication_status']:
        st.error("Please sign in from the Home page and try again.")
        st.stop()

    st.title("Explore topics")
    st.write("Understand your documents better by exploring topics in each document and across documents.")

    with open('auth/config.yaml') as file:
        auth_config = yaml.load(file, Loader=SafeLoader)

    username = st.session_state["username"]
    try:
        hashed_password = auth_config["credentials"]["usernames"][username]["password"]
    except KeyError as e:
        st.error(f"Error: {e}")

    tenant_id = generate_tenant_id(username, hashed_password)

    with open("style.css", "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


    # Perform API call to Orchestrator to retrieve topic_model
    if st.button("Explore topics"):
        st.write("Exploring topics...")
        url = config["orchestrator_url_topics"]
        response = requests.get(url, stream=True)
        print(response)
        if response.status_code == 200:
            # Save the file in chunks
            with open("topic_cache", "wb") as f:
                # for chunk in response.iter_content(chunk_size=4096):  # Adjust chunk_size as needed
                #     if chunk:
                #         f.write(chunk)
                f.write(response.content)
            st.write("Topic model retrieved successfully.")
        else:
            st.write("Error retrieving topic model.")

        

        loaded_model = BERTopic.load("topic_cache", embedding_model=SentenceTransformer("all-MiniLM-L6-v2"))
        st.write(loaded_model.get_topic_info())
        print(loaded_model.get_topic_info())



except Exception as e:
    print("Actual error:", e)
    st.error(f"Please sign in from the Home page and try again. Error: {e}")
    st.stop()