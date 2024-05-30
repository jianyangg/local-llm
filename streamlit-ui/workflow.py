from langgraph.graph import END, StateGraph
from GraphState import GraphState
from Decision import Decision

def buildChatApp():
    """
    This function sets up and compiles the workflow for the custom RAG + LLM workflow.

    Workflow passes around a GraphState object.

    Args:
        st (streamlit): The streamlit object

    Returns:
        StateGraph: The compiled workflow
    """
   
    # Initialise the decision class
    decision = Decision()

    # Intialise the graph
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("give_up", decision.give_up)  # give up if no relevant documents are found.
    workflow.add_node("retrieve_documents", decision.retrieve_documents)  # retrieve documents
    workflow.add_node("grade_documents", decision.grade_documents)  # grade documents
    workflow.add_node("generate", decision.generate)  # generate

    """
    Construct the graph.
    """

    ## 1. Entry Point / Initial Routing: 
    ##    redirects to either "give_up" node or "retrieve_documents" node
    workflow.set_conditional_entry_point(
        decision.initial_routing,
        # Format: <output_from_intialRoutingFn: node_name>
        {
            "give up": "give_up",
            "vectorstore": "retrieve_documents",
        },
    )


    ## 2a. Retrieve Documents
    workflow.add_edge("retrieve_documents", "grade_documents")
    workflow.add_conditional_edges(
       "grade_documents",
        decision.check_relevance,
        {
            "give up": "give_up",
            "generate": "generate",
        },     
    )

    ## 2b. Give Up
    ### the node implicitly prints the message,
    ### asking user to improve the prompt or upload relevant documents
    ### give_up node links directly to the END node
    workflow.add_edge("give_up", END)


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
            "not useful": "give_up",
        },
    )

    chat_app = workflow.compile()

    return chat_app

    