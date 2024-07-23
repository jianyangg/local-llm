import os
import streamlit as st
import yaml
from utils import generate_tenant_id
from yaml.loader import SafeLoader
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from utils import load_docs_from_jsonl, run_topic_model

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

    path = f"topic_models_cache/{tenant_id}"
    if os.path.exists(path) and os.path.exists(f"{path}/topic_model"):
        button_text = "Regenerate topics"
        # Show topics
        loaded_model = BERTopic.load(f"{path}/topic_model", embedding_model=SentenceTransformer("all-MiniLM-L6-v2"))
        st.subheader("Topics Info")
        st.write(loaded_model.get_topic_info())
        st.subheader("Visualize Topics")
        st.write(loaded_model.visualize_topics())
        # st.link_button("View topic graph", "http://localhost:7475")
    else:
        button_text = "Generate topics"

    if st.button(button_text):
        with st.spinner("Running topic model..."):
            # Collate all doc splits
            doc_splits = []
            # All splits are stored in documents/*.jsonl
            # doc_count helps with processing the documents in the same order at the orchestrator side
            doc_count = 0
            for file in os.listdir(f"documents/{tenant_id}"):
                if file.endswith(".jsonl"):
                    doc_splits.extend(load_docs_from_jsonl(f"documents/{tenant_id}/{doc_count}_{file}"))
                    doc_count += 1

            # Perform topic modelling (implicitly saves topic model)
            run_topic_model(doc_splits, tenant_id)
            st.rerun()

except Exception as e:
    print("Actual error:", e)
    st.error(f"Please sign in from the Home page and try again. Error: {e}")
    st.stop()