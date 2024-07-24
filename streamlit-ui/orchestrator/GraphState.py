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
    qn_for_retrieval: str
    generation: str
    give_up: str
    documents: List[str]
    chat_history: list
    attempts: int
    feedback: str
    verdict: str
    prev_ans: str

    # initialise documents as empty list
    documents = []
    chat_history = []
    attempts = 0
