import streamlit as st
import networkx as nx
import json
from pyvis.network import Network
import streamlit.components.v1 as components

# Set the page title and icon
st.set_page_config(page_title="NetworkX Graph Visualization", page_icon=":bar_chart:")

# Load the graph data from JSON file
with open("graph.json", "r") as f:
    data = json.load(f)

# Create a NetworkX graph from the loaded data
G = nx.node_link_graph(data)

# Create a Pyvis Network graph
net = Network(notebook=True)

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

# Save the network to an HTML file
net.show("graph.html")

# CSS to remove borders
st.markdown(
    """
    <style>
    .stApp {
        padding: 0;
        margin: 0;
    }
    iframe {
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a description of the app
st.write("This app visualizes a graph using NetworkX and Pyvis. You can customize the graph using the options in the sidebar.")

# Display the graph in Streamlit
st.title("NetworkX Graph Visualization")

# Read the HTML file
with open("graph.html", "r", encoding='utf-8') as html_file:
    source_code = html_file.read()

# Display the graph using Streamlit's components.html function
components.html(source_code, height=600, width=800)

# Add a footer with additional information
st.markdown("---")
st.markdown("Created by [Your Name](https://yourwebsite.com)")
