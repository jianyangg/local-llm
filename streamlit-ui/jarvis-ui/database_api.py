import os
from app_config import config
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.text_splitter import SemanticChunker
from llmsherpa.readers import LayoutPDFReader
from utils import draw_bounding_box_on_pdf_image

## Load embeddings
embeddings = OllamaEmbeddings(
    base_url=config["ollama_base_url"],	
    model=config["llm_name"]
)

# def docParser(file_path, st):
#     """
#     Parses a document file and returns a split up version of the document.
#     Requires file type to be reflected in the file extension.

#     Args:
#         file_path (str): The path to the document file

#     Returns:
#         List of LangchainDocument objects
#     """

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=20)
#     file_name = os.path.basename(file_path)

#     # This assumes that we can tell file type from file extension
#     # May not work for linux based systems
#     # TODO: NOT THE BEST CHUNKING STRATEGY
#     if file_path.endswith('.docx'):
#         doc_splits = []
#         doc = docxDocument(file_path)

#         para_num = 0
#         for para in doc.paragraphs:
#             if not para.text:
#                 continue

#             langchain_doc_splits = LangchainDocument(
#                 page_content=para.text,
#                 metadata={
#                     "source": file_path,
#                     "chunk_number": para_num,
#                     "chunk_type": "para"
#                 }
#             )
#             doc_splits.append(langchain_doc_splits)
#             para_num += 1

#         return doc_splits

#     else:
#         # Use LLMSherpa for all other file types
#         # If exception occurs, raise it

#         try:
#             # LLMSherpa loader (requires container nlm-ingest to be running)
#             # TODO: Include more metadata into each chunk (i.e., which page is the chunk from)
#             loader = LLMSherpaFileLoader(
#                 file_path=file_path,
#                 new_indent_parser=True,
#                 apply_ocr=True,
#                 strategy="text", # this can be "chunks" or "html".
#                 llmsherpa_api_url="http://nlm-ingestor:5001/api/parseDocument?renderFormat=all",
#             )

#             st.toast(f"Chunking document {file_name}...")
#             docs = loader.load()

#             doc_splits = text_splitter.split_documents(docs)

#             return doc_splits
            
#         except Exception as e:
#             st.write(f"Error: {e}")

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

def find_leaf_nodes(node, leaf_nodes=[]):

    if len(node.children) == 0:
        leaf_nodes.append(node)
    for child in node.children:
        find_leaf_nodes(child, leaf_nodes)

    return leaf_nodes

def docParser(file_path, st, tenant_id):
    layout_root = None
    try:
        reader = LayoutPDFReader(config["nlm_url"])
        try:
            parsed_doc = reader.read_pdf(file_path)
        except FileNotFoundError:
            st.error(f"File {file_path} not found.")
            return []
        layout_root = parsed_doc.root_node
    except Exception as e:
        st.error("Error:", e)

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
        
        # Visualise chunking
        for doc in docs:
            draw_bounding_box_on_pdf_image(doc.metadata, colour="green", location=f"chunks/{tenant_id}")


    return docs

# Upload files
def upload_files(uploaded_files, st, tenant_id, username=config["neo4j_username"], password=config["neo4j_password"]):
    
    combined_doc_splits = []
    for uploaded_file in uploaded_files:
        doc_path = f"documents/{tenant_id}/{uploaded_file.name}"
        st.toast(f"Processing {uploaded_file.name}")
        # check if file exists
        if not os.path.exists(doc_path):
            st.error(f"File {uploaded_file.name} does not exist in {doc_path}.")
            continue
        doc_splits = docParser(doc_path, st, tenant_id)
        print(f"Number of splits: {len(doc_splits)}")
        print("\n")
        combined_doc_splits.extend(doc_splits)

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
            # keyword_index_name="keyword",
            # search_type="hybrid"
        )
        print("Documents written to database")
        return True
    except Exception as e:
        st.error(e.message)
        st.error(e.args)
        return False


