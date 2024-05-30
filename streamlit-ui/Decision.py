from GraphState import GraphState
from CustomLLM import CustomLLM

class Decision:
    def __init__(self):
        self.custom_llm = CustomLLM()

    def initial_routing(self, state: GraphState):
        """
        This routing was initially meant for the LLM to decide between analysis
        and a simple web search. However, since we can't do web search in DSTA,
        we will route between:

        1. Analysis (RAG + LLM)
        2. Retrieval (from Vector Database) -- simple querying of relevant documents
            2a. This can be improved by creating documents that return the actual instance
                of a document relative to just a text-based representation of the documents

        More options can be added in the future. We will explore 1 first.

        Args: 
            st (streamlit): The streamlit object
            graph_state: The state of the graph

        Returns:
            str: Next node to call
        """
        print("Deciding route...")
        user_prompt = state["question"]
        decision = self.custom_llm.initial_router(user_prompt)
        print("DECISON:", decision)

        if decision["datasource"] == "give up":
            print("Route decided: Give up.")
            return "give up"
        elif decision["datasource"] == "vectorstore":
            print("Route decided: Retrieve documents.")
            # TODO: implement the retrieval of documents
            # return "retrieve_documents"
            return "give up"
        else:
            # Should not reach here
            print("Error: Invalid decision in initial_routing.")
            return "give up"

    def give_up(self, state: GraphState):
        print("Give up.")
        print("---" * 10)

        return {"generation": "I'm not capable of answering the question due to insufficient knowledge or your request. \
                Please improve your prompt or upload relevant documents."}

    def retrieve_documents(self):
        return

    def grade_documents(self):
        return

    def generate(self):
        return

    def check_relevance(self):
        return

    def grade_generation_wrt_documents_and_question(self):
        return
