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
        workflow.add_node("retrieve_documents", decision.retrieve_documents)  # retrieve documents
        workflow.add_node("generate", decision.generate_answer)  # generate
        workflow.set_entry_point("retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate")
        workflow.add_edge("generate", END)

    elif chat_mode == "Semantic Search w Agents":
        # Same as Jarvis mode without initial routing

        workflow.add_node("giveup", decision.give_up)  # give up if no relevant documents are found.
        workflow.add_node("retrieve_documents", decision.retrieve_documents)  # retrieve documents
        workflow.add_node("grade_documents", decision.grade_documents)  # grade documents
        workflow.add_node("generate", decision.generate_answer)  # generate
        workflow.add_node("llm", decision.generate_llm_only)  # generate

        workflow.set_entry_point("retrieve_documents")
        workflow.add_edge("retrieve_documents", "grade_documents")

        workflow.add_conditional_edges(
           "grade_documents",
            decision.decide_to_generate,
            {
                "give up": "giveup",
                "generate": "generate",
            },     
        )

        ## 2b. Give Up
        ### the node implicitly prints the message,
        ### asking user to improve the prompt or upload relevant documents
        ### give_up node links directly to the END node
        workflow.add_edge("giveup", END)


        ## 3. Grade Documents (the last few sections of the workflow)
        workflow.add_conditional_edges(
            "generate",
            # this fn checks if generation is relevant to documents
            # and then checks if the generation is useful to answer the question
            decision.grade_generation_wrt_documents_and_question,
            {   
                # not supported by documents
                "not supported": "generate",
                # useful if no hallucinations and answers the question
                "useful": END,
                # not useful in answering the question
                "not useful": "llm",
            },
        )

        workflow.add_edge("llm", END)

    elif chat_mode == "Jarvis":
        # TODO: This is the implementation for a LangGraph workflow.
        # Define the nodes
        workflow.add_node("giveup", decision.give_up)  # give up if no relevant documents are found.
        workflow.add_node("retrieve_documents", decision.retrieve_documents)  # retrieve documents
        workflow.add_node("grade_documents", decision.grade_documents)  # grade documents
        workflow.add_node("generate", decision.generate_answer)  # generate
        workflow.add_node("llm", decision.generate_llm_only)  # generate

        """
        Construct the graph.
        """

        ## 1. Entry Point / Initial Routing: 
        ##    redirects to either "give_up" node or "retrieve_documents" node
        workflow.set_conditional_entry_point(
            decision.initial_routing,
            # Format: <output_from_intialRoutingFn: node_name>
            {
                "generate": "llm",
                "vectorstore": "retrieve_documents",
            },
        )


        ## 2a. Retrieve Documents
        workflow.add_edge("retrieve_documents", "grade_documents")
        workflow.add_conditional_edges(
           "grade_documents",
            decision.decide_to_generate,
            {
                "give up": "giveup",
                "generate": "generate",
            },     
        )

        ## 2b. Give Up
        ### the node implicitly prints the message,
        ### asking user to improve the prompt or upload relevant documents
        ### give_up node links directly to the END node
        workflow.add_edge("giveup", END)


        ## 3. Grade Documents (the last few sections of the workflow)
        workflow.add_conditional_edges(
            "generate",
            # this fn checks if generation is relevant to documents
            # and then checks if the generation is useful to answer the question
            decision.grade_generation_wrt_documents_and_question,
            {   
                # not supported by documents
                "not supported": "generate",
                # useful if no hallucinations and answers the question
                "useful": END,
                # not useful in answering the question
                "not useful": "llm",
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

    