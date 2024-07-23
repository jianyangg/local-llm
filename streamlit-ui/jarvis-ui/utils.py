import fitz
import hashlib
import os
from PIL import ImageDraw
from pdf2image import convert_from_path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from  langchain.schema import Document
import json
from typing import Iterable
from langchain_community.llms import Ollama
from app_config import config
import requests
import pickle
from pprint import pprint

llm = Ollama(model="llama3:instruct", temperature=0, base_url=config["ollama_base_url"], verbose=False)

def generate_hasher(unique_id):
    # Create a new sha256 hash object
    sha_signature = hashlib.sha256(unique_id.encode()).hexdigest()
    return sha_signature

def generate_tenant_id(username, password):
    return generate_hasher(username + password)

def draw_bounding_box_on_pdf_image(doc, dpi=200, colour="red", location="output/"):
    pdf_path = doc["file_path"]
    page_number = doc["page_idx"]
    coordinates = doc["bbox"]

    # Convert PDF page to image
    images = convert_from_path(pdf_path, first_page=page_number + 1, last_page=page_number + 1, dpi=dpi)
    
    # Assuming we have only one image since we specified a single page
    img = images[0]
    
    # Get the size of the PDF page in points (1 point = 1/72 inches)
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc.load_page(page_number)
    page_width, page_height = page.rect.width, page.rect.height
    
    # Scale coordinates to match the image resolution
    scale_x = img.width / page_width
    scale_y = img.height / page_height
    scaled_coordinates = tuple(int(coord * max(scale_x, scale_y)) for coord in coordinates)

    # Manually tune the right limit more
    scaled_coordinates = (scaled_coordinates[0], scaled_coordinates[1], scaled_coordinates[2] + 60, scaled_coordinates[3])
    
    # Draw the bounding box on the image
    draw = ImageDraw.Draw(img)
    draw.rectangle(scaled_coordinates, outline=colour, width=2)

    # Create dir if it doesn't exist
    if not os.path.exists(location):
        os.makedirs(location)

    # Image file name
    image_path = f"{location}/{pdf_path.split('/')[-1].replace('.pdf', f'_page_{page_number}_{coordinates[1]}.png')}"
    
    # Save the image with the bounding box
    img.save(image_path)

    return image_path

def delete_screenshots(tenant_id):
    if not os.path.exists(f"output/{tenant_id}"):
        os.makedirs(f"output/{tenant_id}")
        return
    # delete all the png files in output dir
    for file in os.listdir(f"output/{tenant_id}"):
        if file.endswith(".png"):
            os.remove(f"output/{tenant_id}/{file}")

### For Topic Modelling
def run_topic_model(docs, tenant_id):
    # Might have some issues loading stopwords in Docker
    # nltk.download('stopwords')

    # Remove stop words
    docs_str = []
    stop_words = set(stopwords.words('english'))
    for doc in docs:
        # Remove stop words
        # Only referencing the page_content
        text = doc.page_content
        tokens = word_tokenize(text)
        temp_filtered_text = [word for word in tokens if word.casefold() not in stop_words]
        # ignore all instances of the word Metadata or Content
        temp_filtered_text = [word for word in temp_filtered_text if word.casefold() not in ["metadata", "content"]]
        filtered_text = " ".join(temp_filtered_text)
        docs_str.append(filtered_text)

    # Save docs_str in a jsonl file
    docs_str_path = f"documents/{tenant_id}/collated_docs_str_nlp.pkl"
    with open(docs_str_path, 'wb') as file:
        pickle.dump(docs_str, file)
    print(f"Collated docs_str saved at: {docs_str_path}")

    # Upload collated docs_str
    response = upload_file(docs_str_path)
    print(f"File upload for docs_str response: {response.json()}")

    # Initialise BERTopic
    # Step 1 - Extract embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine')

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english")

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer()

    # Step 6 - (Optional) Fine-tune topic representations with 
    # a `bertopic.representation` model
    representation_model = KeyBERTInspired()

    # All steps together
    topic_model = BERTopic(
        min_topic_size=3,                          # Minimum size of the topic
        embedding_model=embedding_model,          # Step 1 - Extract embeddings
        umap_model=umap_model,                    # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
        representation_model=representation_model # Step 6 - (Optional) Fine-tune topic represenations
    )

    topics, _ = topic_model.fit_transform(docs_str)

    # Generate topic name
    topic_name = {}
    for i in range(len(topic_model.get_topic_info())):
        rep_words = topic_model.get_topic_info()["Representation"][i]
        topic_name[i-1] = llm.invoke(f"Generate a 2 to 3 word TOPIC NAME from the words it's represented by: ({rep_words[:4]}). IMPORTANT: EXCLUDE ANY PREAMBLE OR EXPLANATION.")

    topic_model.set_topic_labels(topic_name)

    # Make directory if it doesn't exist
    path = f"topic_models_cache/{tenant_id}/topic_model.pkl"
    if not os.path.exists(path):
        os.makedirs(path)

    topic_model.save(path, serialization="pickle", save_embedding_model=False, save_ctfidf=True)

    # Upload topic model
    response = upload_file(path)
    print(f"File upload for topic model response: {response.json()}")


    # For plotting of topic-knowledge graph to neo4j
    # uri = "neo4j://localhost:7688"
    # with GraphDatabase.driver(uri, auth=("neo4j", "password")) as driver:
    #     # Clear database
    #     driver.execute_query("MATCH (n) DETACH DELETE n")
    #     # Add all topics
    #     for topic in topic_name:
    #         driver.execute_query(f"CREATE (t:Topic {{name: '{topic_name[topic]}', id: '{topic}'}})")
    #     # Add all documents
    #     doc_path = [doc.metadata['file_path'].split("/")[-1] for doc in docs]
    #     set_doc_path = set(doc_path)
    #     for doc in set_doc_path:
    #         driver.execute_query(f"CREATE (d:Document {{name: '{doc}'}})")
    #     # Add all document chunks
    #     for i, doc in enumerate(docs):
    #         driver.execute_query(f"CREATE (c:Chunk {{id: 'chunk_{i}', text: '{doc.page_content}', name: '{doc.metadata['file_path']}', topic: '{topic_model.get_document_info(docs_str)['Topic'][i]}'}})")
    #     # Add relationships between documents and chunks
    #     for i, doc in enumerate(docs):
    #         driver.execute_query(f"MATCH (d:Document {{name: '{doc.metadata['file_path'].split('/')[-1]}'}}), (c:Chunk {{id: 'chunk_{i}'}}) CREATE (d)-[:HAS_CHUNK]->(c)")
    #         # Add relationships between chunks and topics
    #         driver.execute_query(f"MATCH (c:Chunk {{id: 'chunk_{i}'}}), (t:Topic {{id: '{topic_model.get_document_info(docs_str)['Topic'][i]}'}}) CREATE (c)-[:HAS_TOPIC]->(t)")
    #     # Add direct link between documents and topics
    #     for doc_path in set_doc_path:
    #         # Each main doc has multiple topics associated with it
    #         # Find all topics associated with the doc
    #         chunks = [doc for doc in docs if doc.metadata['file_path'].split('/')[-1] == doc_path]
    #         topics = [topic_model.get_document_info(docs_str)['Topic'][docs.index(chunk)] for chunk in chunks]
    #         set_topics = set(topics)
    #         for topic in set_topics:
    #             driver.execute_query(f"MATCH (d:Document {{name: '{doc_path}'}}), (t:Topic {{id: '{topic}'}}) CREATE (d)-[:HAS_TOPIC]->(t)")

def upload_file(path):
    url = config["orchestrator_url_upload_file"]
    # Assume path from and path to are the same
    assert os.path.exists(path), f"File not found: {path}"
    with open(path, 'rb') as file:
        response = requests.post(url, files={'file': file}, data={'save_path': path})
    return response

### For storing langchain documents
def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')
            
def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

