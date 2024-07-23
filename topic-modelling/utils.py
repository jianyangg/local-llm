from langchain.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from llmsherpa.readers import LayoutPDFReader

config = {
    "ollama_base_url": "http://localhost:11434",
    "llm_name": "ollama",
    "nlm_url": "http://localhost:5001",
}

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
        print("REMOVE:", collated_pg_content[0])

        # Convert to Langchain documents
        docs = [LangchainDocument(page_content=collated_pg_content[i], metadata={key: leaf_nodes[i].block_json[key] for key in ('bbox', 'page_idx', 'level')} | {"file_path": file_path}) for i in range(len(collated_pg_content))]

    return docs

# Define some helper functions
def get_chunks_from_topic(topic_id, topic_model, docs_str, docs):
    """
    docs_str - list of strings
    docs - list of Langchain Document objects
    """
    temp = topic_model.get_document_info(docs_str)["Topic"] == topic_id
    # print(temp)
    df = topic_model.get_document_info(docs_str)[temp]
    # get list of all index
    doc_index = df.index.tolist()
    return [docs[i] for i in doc_index]

def get_chunks_from_query(query, topic_model, docs_str, docs):
    topics = topic_model.find_topics(query)
    print(topics)
    chunks = []
    
    # Select all topics with probability > 0.5
    for i in range(len(topics[1])):
        if topics[1][i] > 0.5:
            print("Topic Chosen:", topics[0][i], "Probability:", topics[1][i])
            chunks.extend(get_chunks_from_topic(topics[0][i], topic_model, docs_str, docs))

    if len(chunks) > 3:
        return chunks
    
    # If not enough chunks, get the top 3 topics
    for i in range(len(chunks), 3):
        print("Getting topic:", i)
        chunks.extend(get_chunks_from_topic(i, topic_model, docs_str, docs))
        if len(chunks) > 3:
            break

    return chunks