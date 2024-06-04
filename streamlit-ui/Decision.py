from GraphState import GraphState
from CustomLLM import CustomLLM
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
# from chromadb import HttpClient
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import OllamaEmbeddings

class Decision:
    def __init__(self):

        # TODO: Temporary code
        # to be replaced with Neo4J implementation and a portal for uploading
        # documents in the vector database
        # urls = [
        #     "https://lilianweng.github.io/posts/2023-06-23-agent/",
        #     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        #     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        # ]

        # docs = [WebBaseLoader(url, requests_per_second=1).load() for url in urls]
        # docs_list = [item for sublist in docs for item in sublist]

        # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #     chunk_size=250, chunk_overlap=0
        # )
        # doc_splits = text_splitter.split_documents(docs_list)

        # model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
        # gpt4all_kwargs = {'allow_download': 'True'}
        # embeddings = GPT4AllEmbeddings(
        #     model_name=model_name,
        #     gpt4all_kwargs=gpt4all_kwargs
        # )

        # # Add to vectorDB
        # vectorstore = Chroma.from_documents(
        #     documents=doc_splits,
        #     collection_name="rag-chroma",
        #     embedding=embeddings,
        # )

        neo4j_config={
            "ollama_base_url": "http://localhost:11434",
            "llm_name": "llama3",
            "neo4j_url": "bolt://localhost:7687",
            "neo4j_username": "neo4j",
            "neo4j_password": "password",	
            "index_name": "parsers_trial_2",
            "node_label": "parsersTrial2"
        }

        # load embedding model
        embeddings = OllamaEmbeddings(
            base_url=neo4j_config["ollama_base_url"],	
            model=neo4j_config["llm_name"]
        )

        # reference document_parsing notebook
        self.vectorstore = Neo4jVector.from_existing_index(
            embeddings,
            url=neo4j_config["neo4j_url"],
            username=neo4j_config["neo4j_username"],
            password=neo4j_config["neo4j_password"],
            index_name=neo4j_config["index_name"],
            # node_label=neo4j_config["node_label"],
            # embedding_node_property="embedding",
            # text_node_properties="text",
        )

        self.retriever = self.vectorstore.as_retriever()
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

        if decision["datasource"] == "generate":
            print("Route decided: Generate answer without RAG.")
            return "generate"
        elif decision["datasource"] == "vectorstore":
            print("Route decided: Retrieve documents.")
            # TODO: implement the retrieval of documents
            return "vectorstore"
        else:
            # Should not reach here
            print("Error: Invalid decision in initial_routing.")
            return "giveup"


    def give_up(self, state: GraphState):
        """
        This function is called when the LLM decides to give up on the question.
        We can change this function to do something else in the future.
        i.e., web search, etc.

        Args:
            state (GraphState): The state of the graph

        Returns:
            dict: The response to the user
        """
        print("Give up.")
        print("---" * 10)

        return {"generation": "I'm not capable of answering the question due to insufficient knowledge or your request. \
                Please improve your prompt or upload relevant documents."}


    def retrieve_documents(self, state):
        """
        Retrieve documents from vectorstore
        TODO: To be replaced with Neo4J database

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("Retrieving documents...")
        # get the question from the state which is a child of GraphState
        question = state["question"]

        # Retrieval
        ## use our vector db to fetch documents relevant to our question
        # documents = self.retriever.invoke(question, top_k=10)

        # for each document, calculate the score for similarity to the question
        docs_with_score = self.vectorstore.similarity_search_with_score(question, k=10)
        for doc, score in docs_with_score:
            print("-" * 80)
            print("Score: ", score)
            print(doc.page_content)
            print("-" * 80)

        documents = [doc for doc, _ in docs_with_score]
        return {"documents": documents, "question": question}


    def grade_documents(self, state):
        """
        Grade the relevance of documents based on the question.
        # TODO: Instead of this, we can also consider the relative nearness of these documents
        # to the question. This can be done by using the vectordatabase.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updated state with relevant documents, question and give_up status
        """
        print("Grading documents...")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        relevant_docs = 0
        for d in documents:
            score = self.custom_llm.retrieval_grader(
                question, d.page_content
            )
            ## recall that the score variable holds a json {"score": "yes"}
            grade = score["score"]
            # Document relevant
            if grade.lower() == "yes":
                print("Document relevant.")
                filtered_docs.append(d)
                relevant_docs += 1

            else:
                print("Document not relevant.")

        # If no relevant documents, we give up.
        if relevant_docs == 0:
            print("No relevant documents found.")
            give_up = "yes"
        else:
            give_up = "no"

        # give_up status
        return {"documents": filtered_docs, "question": question, "give_up": give_up}


    def generate_answer(self, state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("Generating answer...")
        question = state["question"]
        documents = state["documents"]

        # Concatenate all documents
        if documents:
            context = "\n\n".join(doc.page_content for doc in documents)
        else:
            context = "No relevant documents found. Answer solely based off the prompt given."

        generated_answer = self.custom_llm.answer_generator(context=context, qn=question)
        # Only the last item is generated in this function
        return {"documents": documents, "question": question, "generation": generated_answer}


    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or give up
        Criteria is based on the relevance of the documents to the question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("Conditional Edge: Deciding to generate...")
        give_up = state["give_up"]

        if give_up == "yes":
            print("Decision: Give up.")
            return "give up"
        else:
            # We have relevant documents, so generate answer
            print("Decision: Generate answer.")
            return "generate"



    def grade_generation_wrt_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("Checking hallucinations and question-answering...")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        # if no documents, we can't grade the generation w.r.t. documents
        # so run answer_grader only
        if not documents:
            print("No documents to grade.")
            score = self.custom_llm.answer_grader(generated_answer=generation, qn=question)
            grade = score["score"]
            if grade == "yes":
                print("Verdict: Generated answer is useful.")
                return "useful"
            else:
                print("Verdict: Generated answer is not useful.")
                return "not useful"

        score = self.custom_llm.hallucination_grader(
            generated_answer=generation, documents=documents
        )

        # TODO: Consider passing the grade as a boolean instead of a string.
        grade = score["score"]

        # Check hallucination as in whether generation is grounded in documents
        if grade == "yes":
            print("Verdict: Generated answer is grounded in documents.")
            # Check question-answering
            print("Checking if generation answers the question...")
            score = self.custom_llm.answer_grader(generated_answer=generation, qn=question)
            grade = score["score"]
            if grade == "yes":
                print("Verdict: Generated answer is useful.")
                return "useful"
            else:
                print("Verdict: Generated answer is not useful.")
                return "not useful"
        else:
            print("Verdict: Generated answer is not grounded in documents.")
            return "not supported"
