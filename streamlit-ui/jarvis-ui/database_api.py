import os
from app_config import config
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.text_splitter import SemanticChunker
from llmsherpa.readers import LayoutPDFReader
from utils import draw_bounding_box_on_pdf_image, save_docs_to_jsonl, upload_file

## Load embeddings
# TODO: Try using sentence-transformers instead of Ollama
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = OllamaEmbeddings(
    base_url=config["ollama_base_url"],	
    model=config["llm_name"]
)

def parent_chain(node):
    """
    Returns the parent chain of the block consisting of all the parents of the block until the root.
    """
    chain = []
    parent = node.parent
    while parent:
        chain.append(parent)
        parent = parent.parent
    chain.reverse()
    return chain

def parent_text(node):
    """
    Returns the text of the parent chain of the block. This is useful for adding section information to the text.
    """
    chain = parent_chain(node)
    header_texts = []
    para_texts = []
    for p in chain:
        if p.tag == "header":
            header_texts.append(p.to_text()) 
        elif p.tag in ['list_item', 'para']:
            para_texts.append(p.to_text())
    text = "\n>\n".join(header_texts)
    if len(para_texts) > 0:
        text +="\n\n".join(para_texts)
    return text
   
def to_context_text(node, include_section_info=True):
    """
    This is a customised function largely derived from layout_reader.py of the llmsherpa library
    Returns the text of the block with section information. This provides context to the text.
    """
    text = "Metadata:\n"
    if include_section_info and parent_text(node) != "":
        text += parent_text(node) + "  >\n"
    text += "Content:\n"
    if node.tag in ['list_item', 'para']:
        text += node.to_text(include_children=True, recurse=True)
    elif node.tag == 'table':
        text += node.to_html()
    else:
        text += node.to_text(include_children=True, recurse=True)
    return text

def is_use_semantic_chunking(leaf_nodes):
    # Returns true if more than 50% of paragraphs have only one line
    count = 0
    num_paras = len([node for node in leaf_nodes if node.tag == "para"])
    
    for node in leaf_nodes:
        if node.tag == "para":
            txt = node.to_text().strip()

            lines = txt.split("\n")
            if len(lines) == 1:
                count += 1
                # print("Single line para:", txt)

    print("Number of single line para:", count)
    print("Number of paragraphs:", num_paras)
    return count > num_paras/2

def find_leaf_nodes(node, leaf_nodes=None):
    if leaf_nodes is None:
        leaf_nodes = []

    if len(node.children) == 0:
        leaf_nodes.append(node)
    for child in node.children:
        find_leaf_nodes(child, leaf_nodes)

    return leaf_nodes


def docParser(file_path, st=None, tenant_id=None, visualise_chunking=False):
    layout_root = None

    try:
        reader = LayoutPDFReader(config["nlm_url"])
        try:
            print("Reading file:", file_path)
            parsed_doc = reader.read_pdf(file_path)
        except FileNotFoundError:
            if st is not None:
                st.error(f"File {file_path} not found.")
            print(f"File {file_path} not found.")
            return []
        layout_root = parsed_doc.root_node
    except Exception as e:
        if st is not None:
            st.error("Error:", e)
        print("Error:", e)

    leaf_nodes = find_leaf_nodes(layout_root)

    if is_use_semantic_chunking(leaf_nodes):
        # Perform semantic chunking

        ## Load embeddings
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"],	
            model=config["llm_name"]
        )
        ## Chunk documents using semantic chunker
        text_splitter = SemanticChunker(
            embeddings, breakpoint_threshold_type="percentile"
        )

        full_text = ""
        for child in layout_root.children:
            full_text += child.to_text(include_children=True, recurse=True) + "\n"

        docs = text_splitter.create_documents([full_text])
    else:
        print("Using llmsherpa")
        # Use chunks from llmsherpa
        # Each chunk is each leaf_node with to_context_text()
        collated_pg_content = [to_context_text(node) for node in leaf_nodes]

        # Convert to Langchain documents
        docs = [LangchainDocument(page_content=collated_pg_content[i], metadata={key: leaf_nodes[i].block_json[key] for key in ('bbox', 'page_idx', 'level')} | {"file_path": file_path}) for i in range(len(collated_pg_content))]
        
        if visualise_chunking:
            # Visualise chunking
            for doc in docs:
                draw_bounding_box_on_pdf_image(doc.metadata, colour="green", location=f"chunks/{tenant_id}")

    return docs

# Upload files
def upload_files(uploaded_files, st, tenant_id, username=config["neo4j_username"], password=config["neo4j_password"], get_topics=False):
    
    combined_doc_splits = []
    existing_files = os.listdir(f"documents/{tenant_id}")
    # doc_count works as the document id
    curr_doc_count = max([int(file.split("_")[0]) for file in existing_files if file.endswith(".jsonl")], default=0)
    print("Current document count:", curr_doc_count)
    for uploaded_file in uploaded_files:
        # Remove all json files that are in the uploaded_files
        # The jsonl file has a <number>_<filename>.jsonl format
        for file in existing_files:
            if file.endswith(f"{uploaded_file.name}.jsonl") and len(file.split("_")) == len(uploaded_file.name.split("_")) + 1:
                os.remove(f"documents/{tenant_id}/{file}")
                print(f"Removed {file}")

        doc_path = f"documents/{tenant_id}/{uploaded_file.name}"
        st.toast(f"Processing {uploaded_file.name}")
        # check if file exists
        if not os.path.exists(doc_path):
            st.error(f"File {uploaded_file.name} does not exist in {doc_path}.")
            continue
        doc_splits = docParser(doc_path, st, tenant_id)

        # Save as jsonl
        jsonl_path = f"documents/{tenant_id}/{curr_doc_count}_{uploaded_file.name}.jsonl"
        curr_doc_count += 1

        # Store doc_splits
        save_docs_to_jsonl(doc_splits, jsonl_path)

        # Upload jsonl file to orchestrator
        response = upload_file(jsonl_path)
        print(f"File upload for {upload_file} doc splits response: {response.json()}")

        print(f"Number of splits: {len(doc_splits)}")
        print("\n")
        combined_doc_splits.extend(doc_splits)

    # Add Source: to each chunk for better context
    for doc in combined_doc_splits:
        doc.page_content = f"Source: {doc.metadata['file_path']}\n\n{doc.page_content}"

    print("Writing to database in index:", tenant_id)
    try:
        # stores the parsed documents in the Neo4j database
        st.toast("Writing to database...")
        Neo4jVector.from_documents(
            documents=combined_doc_splits,
            embedding=embeddings,
            url=config["neo4j_url"],
            username=username,
            password=password,
            index_name=tenant_id,
            node_label=tenant_id,
            keyword_index_name="keyword",
            search_type="hybrid"
        )
        print("Documents written to database")
        return True
    except Exception as e:
        st.error(e.message)
        st.error(e.args)
        return False


