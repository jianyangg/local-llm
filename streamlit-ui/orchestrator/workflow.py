from langgraph.graph import END, StateGraph
from GraphState import GraphState
from Decision import Decision

def buildChatApp(tenant_id, chat_mode):
    """
    This function sets up and compiles the workflow for the custom RAG + LLM workflow.

    Workflow passes around a GraphState object.

    Args:
        tenant_id (str): The tenant ID
        chat_mode (str): The chat mode ie All-Purpose, RAG, Chatbot

    Returns:
        StateGraph: The compiled workflow
    """
   
    # Initialise the decision class
    decision = Decision(tenant_id)

    # Intialise the graph
    workflow = StateGraph(GraphState)

    if chat_mode == "Standard RAG":
        print("Building Standard RAG")
        # Based off chat history, rephrase question to include more context
        workflow.add_node("rephrase_question", decision.rephrase_question)
        workflow.set_entry_point("rephrase_question")
        workflow.add_edge("rephrase_question", "retrieve_documents")
        workflow.add_node("retrieve_documents", decision.retrieve_documents)  # retrieve documents
        workflow.add_node("generate", decision.generate_answer)  # generate
        workflow.add_edge("retrieve_documents", "generate")
        workflow.add_edge("generate", END)

    elif chat_mode == "Jarvis":
        """
        Construct the graph.
        """
        workflow.set_entry_point("rephrase_question")
        # Retrieve documents after rephrasing the question
        workflow.add_edge("rephrase_question", "retrieve_documents")
        # Define the nodes
        workflow.add_node("retrieve_documents", decision.retrieve_documents)  # retrieve documents
        workflow.add_node("generate", decision.generate_answer)  # generate ans in RAG format
        workflow.add_node("critic", decision.give_feedback)  # generate feedback
        workflow.add_node("rephrase_question", decision.rephrase_question)
        # Define the edges
        workflow.add_edge("retrieve_documents", "generate")
        workflow.add_edge("generate", "critic") # critic leaves feedback on generation based on relevance to documents first then to answer
        # conditional edge on critic to direct to re-generate or generate without RAG or end
        workflow.add_conditional_edges(
            "critic",
            decision.critic_rerouter, # simply returns the verdict in graph state
            {
                "generate": "generate", # regenerate RAG with feedback, to retry
                "completed": END, # end the conversation, if answer is satisfactory
            },
        )

    elif chat_mode == "Standard Chatbot":
        # The workflow just has the generate node.
        workflow.set_entry_point("llm")
        workflow.add_node("llm", decision.generate_llm_only)
        workflow.add_edge("llm", END)

    else:
        # Should never reach here
        raise ValueError(f"Invalid chat mode: {chat_mode}")

    chat_app = workflow.compile()

    return chat_app

    