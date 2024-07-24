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
    print("Result of finding relevant topics to query:", topics)
    chunks = []
    topic_index = -1

    for i in range(len(topics[1])):
        probs = topics[1][i]
        # Probabilities are sorted in descending order
        if probs < 0.5:
            break
        print("Topic Chosen:", topics[0][i], "Probability:", topics[1][i])
        chunks.extend(get_chunks_from_topic(topics[0][i], topic_model, docs_str, docs))
        topic_index = i

    # We assume that we need at least 3 chunks from our query for topic modelling
    if len(chunks) > 3:
        return chunks
    
    print("Not enough chunks. Getting additional chunks from less relevant topics.")
    # Keep iterating through the leftover topics until 
    for i in range(topic_index+1, len(topics[1])):
        print("Getting less relevant topic:", topics[0][i], "Probability:", topics[1][i])
        chunks.extend(get_chunks_from_topic(topics[0][i], topic_model, docs_str, docs))
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
