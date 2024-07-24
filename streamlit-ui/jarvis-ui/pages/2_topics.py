import os
import streamlit as st
import yaml
from utils import generate_tenant_id
from yaml.loader import SafeLoader
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from utils import load_docs_from_jsonl, run_topic_model
import pandas as pd
import json
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

def get_topic_info_graph(tenant_id):
    # Load the graph data from JSON file
    with open(f"topic_models_cache/{tenant_id}/graph.json", "r") as f:
        data = json.load(f)

    # Create a NetworkX graph from the loaded data
    G = nx.node_link_graph(data, directed=True)

    # Create a Pyvis Network graph
    net = Network(notebook=True, directed=True)

    # Set the physics options to reduce elasticity
    net.set_options("""
    var options = {
    "nodes": {
        "font": {
        "size": 16
        }
    },
    "physics": {
        "barnesHut": {
        "gravitationalConstant": -2000,
        "centralGravity": 0.3,
        "springLength": 300,
        "springConstant": 0.01,
        "damping": 0.09
        },
        "minVelocity": 0.75
    }
    }
    """)

    net.from_nx(G)

    html_path = f"topic_models_cache/{tenant_id}/graph.html"

    # Save the network to an HTML file
    net.show(html_path)

    # Read the HTML file
    with open(html_path, "r", encoding='utf-8') as html_file:
        source_code = html_file.read()

    # Display the graph using Streamlit's components.html function
    components.html(source_code, height=600, width=800)


try:
    st.set_page_config(page_title="Topics", page_icon=":bar_chart:")

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
    if os.path.exists(path) and os.path.exists(f"{path}/topic_model.pkl"):
        button_text = "Regenerate topics"
    else:
        button_text = "Generate topics"

    if st.button(button_text):
        with st.spinner("Running topic model..."):
            # Collate all doc splits
            doc_splits = []
            # All splits are stored in documents/*.jsonl
            for file in os.listdir(f"documents/{tenant_id}"):
                if file.endswith(".jsonl"):
                    doc_splits.extend(load_docs_from_jsonl(f"documents/{tenant_id}/{file}"))

            # Perform topic modelling (implicitly saves topic model)
            run_topic_model(doc_splits, tenant_id)
            st.rerun()

    if button_text == "Regenerate topics":
        
        tab1, tab2 = st.tabs(["Topic Info Graph", "Topic Info Table"])

        with tab1:
            with st.spinner("Loading topic info graph. Please wait..."):
                get_topic_info_graph(tenant_id)

        with tab2:
            try:
                topic_info_table_df = pd.read_pickle(f"topic_models_cache/{tenant_id}/topic_info_table.pkl")
                st.dataframe(topic_info_table_df)
            except FileNotFoundError:
                st.error("The topic info table has not been generated yet. Please generate it first.")
            except Exception as e:
                st.error(f"An error occurred while loading the topic info table. Error: {e}")


except Exception as e:
    print("Actual error:", e)
    st.error(f"Please sign in from the Home page and try again or contact the administrator.")
    st.stop()

