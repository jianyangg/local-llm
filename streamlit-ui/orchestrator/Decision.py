from GraphState import GraphState
from CustomLLM import CustomLLM
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import OllamaEmbeddings
from flashrank import Ranker, RerankRequest
from langchain.docstore.document import Document as LangchainDocument
from app_config import config


class Decision:
    def __init__(self, tenant_id: str):
        # load embedding model
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"],	
            model=config["llm_name"]
        )

        print("Using vector store for tenant id:", tenant_id)

        # TODO: Bad practice to use exceptions as part of logic.
        try: 
            print("Trying to fetch from existing index...")
            # reference document_parsing notebook
            self.vectorstore = Neo4jVector.from_existing_index(
                embeddings,
                url=config["neo4j_url"],
                username=config["neo4j_username"],
                password=config["neo4j_password"],
                index_name=tenant_id,
                node_label=tenant_id,
            )
        except Exception as e:
            print("Error:", e)
            print("Index does not exist. Creating index...")
            self.vectorstore = Neo4jVector.from_documents(
                documents="",
                url=config["neo4j_url"],
                username=config["neo4j_username"],
                password=config["neo4j_password"],
                embedding=embeddings,
                index_name=tenant_id,
                node_label=tenant_id,
            )
            print(f"Index created for {tenant_id}")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 25, 'fetch_k': 50, 'score_threshold': 0.6})
        self.custom_llm = CustomLLM(tenant_id)
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="cache")

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
        chat_history = state["chat_history"]
        decision = self.custom_llm.initial_router(user_prompt, chat_history)
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

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("Retrieving documents...")
        # get the question from the state which is a child of GraphState
        question = state["question"]

        # Retrieval
        # # use our vector db to fetch documents relevant to our question
        # documents = self.retriever.invoke(question, top_k=10)

        # # for each document, calculate the score for similarity to the question
        # docs_with_score = self.vectorstore.similarity_search_with_score(question, k=10)
        # for doc, score in docs_with_score:
        #     print("-" * 80)
        #     print("Score: ", score)
        #     print(doc.page_content)
        #     print("-" * 80)

        # final_docs = [doc for doc, _ in docs_with_score]

        def docs_to_passages(docs):
            idx = 0
            passages = []
            for doc in docs:
                passages.append({
                    "id": idx,
                    "text": doc.page_content,
                    "meta": doc.metadata
                })
                idx += 1
            return passages
        
        def passages_to_langchainDocument(passages):
            docs = []
            for passage in passages:
                docs.append(LangchainDocument(page_content=passage['text'], metadata=passage['meta']))
            return docs
        
        def pretty_print_docs(docs):
            print(
                f"\n{'-' * 100}\n".join(
                    [
                        f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                        for i, d in enumerate(docs)
                    ]
                )
            )

        docs = self.retriever.invoke(question)
        print("Number of preliminary docs retrieved:", len(docs))

        if len(docs) != 0:
            # Using reranker
            rerankrequest = RerankRequest(query=question, passages=docs_to_passages(docs))
            ranked_passages = self.ranker.rerank(rerankrequest)

            print("Number of reranked docs:", len(ranked_passages))
            # Exclude scores below 0.8
            filtered_ranked_passages = [doc for doc in ranked_passages if doc['score'] >= 0.8]
            # If query isn't specific enough, the score will be very low. In this case, we can use the top 5 docs.
            filtered_ranked_passages = filtered_ranked_passages if len(filtered_ranked_passages) > 3 else ranked_passages[:10]
            print("Number of filtered ranked passages:", len(filtered_ranked_passages))
            final_docs = passages_to_langchainDocument(filtered_ranked_passages)

            print(f"Final: Documents retrieved: {len(final_docs)}")
            pretty_print_docs(final_docs)
        else:
            final_docs = []
            print("Final: No documents retrieved.")

        return {"documents": final_docs, "question": question}


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
        chat_history = state["chat_history"]

        # Score each doc
        filtered_docs = []
        relevant_docs = 0
        for d in documents:
            score = self.custom_llm.retrieval_grader(
                question,
                d.page_content,
                chat_history
            )
            ## recall that the score variable holds a json {"score": "yes"}
            print("Score:", score)
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
        return {"documents": filtered_docs, "question": question, "give_up": give_up, "chat_history": chat_history}


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
        chat_history = state["chat_history"]

        # limit due to limited context length of llm
        limit = 7500

        # process documents stored in the compressed_docs
        docs = [doc.page_content for doc in documents]
        context = "\n\n---\n\n".join(docs)

        # remove all text past limit
        context = context[:limit]
        
        print("Context:", context)

        generated_answer = self.custom_llm.answer_generator(context=context, qn=question, chat_history=chat_history)

        # Only the last item is generated in this function
        return {"documents": documents, "question": question, "generation": generated_answer, "chat_history": chat_history}

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
        chat_history = state["chat_history"]
        limit = 7500

        # process documents stored in the compressed_docs
        docs_content = [doc.page_content for doc in documents]
        context = "\n\n---\n\n".join(docs_content)

        # no relevant documents found.
        if not documents:
            print("Verdict: No relevant documents found.")
            return "not useful"

        score = self.custom_llm.hallucination_grader(
            generated_answer=generation,
            documents=context[:limit],
        )

        print("SCORE", score)
        print()
        # TODO: Consider passing the grade as a boolean instead of a string.
        grade = score["score"]

        # Check hallucination as in whether generation is grounded in documents
        if grade == "yes":
            print("Verdict: Generated answer is grounded in documents.")
            # Check question-answering
            print("Checking if generation answers the question...")
            score = self.custom_llm.answer_grader(generated_answer=generation, qn=question, chat_history=chat_history)
            grade = score["score"]
            if grade == "yes":
                print("Verdict: Generated answer is useful.")
                return "useful"
            else:
                print("Verdict: Generated answer is not useful.")
                # Answer grounded in documents but NOT useful
                # Generate answer solely using LLM.
                return "not useful"
        else:
            print("Verdict: Generated answer is not grounded in documents.")
            # Regenerate answer using documents
            return "not supported"
            
    def generate_llm_only(self, state):
        """
        Generate answer without the need for documents.
        
        Args:
            state (dict): The current graph state

        Returns:
            str: The generated answer
        """

        print("Generating answer without RAG...")
        question = state["question"]
        chat_history = state["chat_history"]
        generated_answer = self.custom_llm.llm_only(qn=question, chat_history=chat_history)
        return {"generation": generated_answer}
    

    def rephrase_question(self, state):
        """
        Rephrase the question in relation to the chat_history to improve the quality of the answer.

        Args:
            state (dict): The current graph state

        Returns:
            str: The rephrased question
        """
        print("Rephrasing question to include context...")
        question = state["question"]
        chat_history = state["chat_history"]
        rephrased_question = self.custom_llm.rephraser(question, chat_history)
        return {"question": rephrased_question, "chat_history": chat_history}