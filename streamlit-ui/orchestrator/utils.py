from typing import Iterable
from  langchain.schema import Document
import json

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

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
