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

    # ## TODO: This is a temporary implementation for direct RAG.
    if chat_mode == "Semantic Search w/o Agents":
        print("Building Semantic Search w/o Agents")
        # Based off chat history, rephrase question to include more context
        workflow.add_node("rephrase_question", decision.rephrase_question)
        workflow.set_entry_point("rephrase_question")
        workflow.add_edge("rephrase_question", "retrieve_documents")
        workflow.add_node("retrieve_documents", decision.retrieve_documents)  # retrieve documents
        workflow.add_node("generate", decision.generate_answer)  # generate
        workflow.add_edge("retrieve_documents", "generate")
        workflow.add_edge("generate", END)

    elif chat_mode == "Semantic Search w Agents":
        # Define the nodes
        workflow.add_node("retrieve_documents", decision.retrieve_documents)  # retrieve documents
        workflow.add_node("grade_documents", decision.grade_documents)  # grade documents
        workflow.add_node("generate", decision.generate_answer)  # generate ans in RAG format
        workflow.add_node("llm", decision.generate_llm_only)  # generate and w/o RAG format
        workflow.add_node("critic", decision.give_feedback)  # generate feedback

        """
        Construct the graph.
        """

        workflow.set_entry_point("rephrase_question")

        # Based off chat history, rephrase question to include more context
        workflow.add_node("rephrase_question", decision.rephrase_question)
        workflow.add_edge("rephrase_question", "retrieve_documents")


        ## 2a. Retrieve Documents
        workflow.add_edge("retrieve_documents", "grade_documents")
        workflow.add_conditional_edges(
           "grade_documents",
            decision.decide_to_generate,
            {
                "llm": "llm", # Give up doing RAG since none are relevant, generate with LLM directly. 
                "generate": "generate", # Continue using RAG since relevant documents found
            },     
        )

        workflow.add_edge("generate", "critic") # critic leaves feedback on generation based on relevance to documents first then to answer

        # conditional edge on critic to direct to re-generate or generate without RAG or end
        workflow.add_conditional_edges(
            "critic",
            decision.critic_rerouter, # simply returns the verdict in graph state
            {
                "generate": "generate", # regenerate RAG with feedback, to retry
                "completed": END, # end the conversation, if ans is satisfactory
            },
        )

        workflow.add_edge("llm", END)

    elif chat_mode == "Jarvis":
        # TODO: This is the implementation for a LangGraph workflow.
        # Define the nodes
        workflow.add_node("retrieve_documents", decision.retrieve_documents)  # retrieve documents
        workflow.add_node("grade_documents", decision.grade_documents)  # grade documents
        workflow.add_node("generate", decision.generate_answer)  # generate ans in RAG format
        workflow.add_node("llm", decision.generate_llm_only)  # generate and w/o RAG format
        workflow.add_node("critic", decision.give_feedback)  # generate feedback

        """
        Construct the graph.
        """

        ## 1. Entry Point / Initial Routing: 
        workflow.set_conditional_entry_point(
            decision.initial_routing,
            {
                "generate": "llm",
                "vectorstore": "rephrase_question",
            },
        )

        # Based off chat history, rephrase question to include more context
        workflow.add_node("rephrase_question", decision.rephrase_question)
        workflow.add_edge("rephrase_question", "retrieve_documents")


        ## 2a. Retrieve Documents
        workflow.add_edge("retrieve_documents", "grade_documents")
        workflow.add_conditional_edges(
           "grade_documents",
            decision.decide_to_generate,
            {
                "llm": "llm", # Give up doing RAG since none are relevant, generate with LLM directly. 
                "generate": "generate", # Continue using RAG since relevant documents found
            },     
        )

        workflow.add_edge("generate", "critic") # critic leaves feedback on generation based on relevance to documents first then to answer

        # conditional edge on critic to direct to re-generate or generate without RAG or end
        workflow.add_conditional_edges(
            "critic",
            decision.critic_rerouter, # simply returns the verdict in graph state
            {
                "generate": "generate", # regenerate RAG with feedback, to retry
                "completed": END, # end the conversation, if ans is satisfactory
            },
        )

        workflow.add_edge("llm", END)


    elif chat_mode == "Chatbot":
        # The workflow just has the generate node.
        workflow.add_node("llm", decision.generate_llm_only)
        workflow.set_entry_point("llm")
        workflow.add_edge("llm", END)

    else:
        # Should never reach here
        raise ValueError(f"Invalid chat mode: {chat_mode}")

    chat_app = workflow.compile()

    return chat_app

    