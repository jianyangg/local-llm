from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    give_up: str
    documents: List[str]
    chat_history: list

    # initialise documents as empty list
    documents = []
    chat_history = []
