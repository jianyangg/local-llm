from GraphState import GraphState
from CustomLLM import CustomLLM
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import OllamaEmbeddings
from flashrank import Ranker, RerankRequest
from langchain.docstore.document import Document as LangchainDocument
from app_config import config
from termcolor import cprint
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from utils import get_chunks_from_query, load_docs_from_jsonl
import os
import pickle

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
                keyword_index_name=f"keyword{tenant_id}",
                search_type="hybrid"
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
                keyword_index_name=f"keyword{tenant_id}",
                search_type="hybrid"
            )
            print(f"Index created for {tenant_id}")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 25, 'fetch_k': 50, 'score_threshold': 0.6})
        self.custom_llm = CustomLLM(tenant_id)
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="cache")

        # docs_str: Document chunks page content with stopwords removed
        docs_str_path = f"documents/{tenant_id}/collated_docs_str_nlp.pkl"
        if os.path.exists(docs_str_path):
            with open(docs_str_path, 'rb') as file:
                self.docs_str = pickle.load(file)
                print("Loaded docs_str from cache.", len(self.docs_str))
        else:
            print("No docs_str found.")
            self.docs_str = None

        # doc_splits: Document chunks
        self.docs_splits = []
        if os.path.exists(f"documents/{tenant_id}"):
            doc_splits_paths = [dir for dir in os.listdir(f"documents/{tenant_id}") if dir.endswith(".jsonl")]
            doc_splits_paths.sort()
            for doc_splits_path in doc_splits_paths:
                print(f"Loading doc_splits from {doc_splits_path}")
                temp_docs = load_docs_from_jsonl(f"documents/{tenant_id}/{doc_splits_path}")
                self.docs_splits.extend(temp_docs)
        else:
            print("No doc splits found")
            os.makedirs(f"documents/{tenant_id}", exist_ok=True)

        topic_model_path = f"topic_models_cache/{tenant_id}/topic_model.pkl"
        if os.path.exists(topic_model_path):
            print("Loading topic model...")
            self.topic_model = BERTopic.load(topic_model_path, embedding_model=SentenceTransformer("all-MiniLM-L6-v2"))
            print("Topic model loaded.")
        else:
            print("No topic model found.")
            self.topic_model = None

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

        if decision["datasource"] == "generate":
            print("Route decided: Generate answer without RAG.")
            return "generate"
        if decision["datasource"] == "vectorstore":
            print("Route decided: Retrieve documents.")
            # TODO: implement the retrieval of documents
            return "vectorstore"
        
        print("Error")
        return "generate"


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
        qn_for_retrieval = state["qn_for_retrieval"]

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
        
        # Retrieve documents from vector DB
        docs = self.retriever.invoke(qn_for_retrieval)
        print("Number of docs retrieved from vector db:", len(docs))
        cprint(f"\nExample doc from vectorstore: {docs[0] if len(docs) > 0 else 'No docs'}\n", "green")

        topic_docs = []
        # Add documents retrieved from topic model
        if self.topic_model != None and self.docs_str != None:
            assert len(self.docs_str) == len(self.docs_splits), "Length of docs_str and docs_splits do not match."
            topic_docs = get_chunks_from_query(qn_for_retrieval, self.topic_model, self.docs_str, self.docs_splits)
            print("Number of docs retrieved from topic model:", len(topic_docs))
            cprint(f"\nExample doc from topic_model: {topic_docs[0]}\n", "green")

        docs.extend(topic_docs)

        print("Number of preliminary docs retrieved:", len(docs))

        if len(docs) != 0:
            # Using reranker
            rerankrequest = RerankRequest(query=question, passages=docs_to_passages(docs))
            ranked_passages = self.ranker.rerank(rerankrequest)

            # Exclude scores below 0.8
            # Get top_n number of docs
            top_n = 15
            # sort ranked_passages by score
            sorted_ranked_passages = sorted(ranked_passages, key=lambda x: x['score'], reverse=True)
            filtered_ranked_passages = sorted_ranked_passages[:top_n]
            final_docs = passages_to_langchainDocument(filtered_ranked_passages)
            cprint(f"\nExample doc from reranker: {final_docs[0] if len(final_docs) > 0 else 'No docs'}\n", "green")
            print(f"No. of final documents retrieved: {len(final_docs)}")
            # pretty_print_docs(final_docs)
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

        # Score each doc
        filtered_docs = []
        relevant_docs = 0
        for d in documents:
            formatted_string = f"From:{d.metadata['file_path']}: {d.page_content}"
            score = self.custom_llm.retrieval_grader(
                question,
                formatted_string,
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

        # give_up status and initialise attempts and feedback
        return {"documents": filtered_docs, "question": question, "give_up": give_up, "attempts": 0, "feedback": ""}


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
        feedback = state["feedback"]

        # process documents stored in the compressed_docs
        docs = ["From" + doc.metadata["file_path"] + ": " + doc.page_content for doc in documents]
        context = "\n\n---\n\n".join(docs)
        limit = 2000
        if len(context) > limit:
            context = context[:limit]

        generated_answer = self.custom_llm.answer_generator(context=context, qn=question, chat_history=chat_history, feedback=feedback)

        print()
        cprint(f"Question: {question}", "yellow")
        print()
        cprint(f"Answer: {generated_answer}", "yellow")
        print("---" * 20)

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
            print("Decision: Generate with LLM directly.")
            return "llm"
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

        # process documents stored in the compressed_docs
        docs_content = ["From" + doc.metadata["file_path"] + ": " + doc.page_content for doc in documents]
        context = "\n\n---\n\n".join(docs_content)

        # no relevant documents found.
        if not documents:
            print("Verdict: No relevant documents found.")
            return "not useful"

        score = self.custom_llm.hallucination_grader(
            generated_answer=generation,
            documents=context[:4000], # limit the context to 4000 characters; arbitrary number < 8k
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
        cprint(f"Chat History: {chat_history}", "yellow")
        cprint(f"Question: {question}", "yellow")
        cprint(f"Answer: {generated_answer}", "yellow")
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
        return {"question": question, "chat_history": chat_history, "qn_for_retrieval": rephrased_question}
    

    def give_feedback(self, state):
        # This is on the RAG path.
        # critic grades the generation based on relevance to documents first then to answer
        # critic then leaves feedback on generation based on relevance to documents first then to answer
        # if any of the feedback is negative, the critic will ask for a re-generation
        print("Critic: Grading generation...")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        chat_history = state["chat_history"]
        feedback = state["feedback"]
        attempts = state["attempts"]

        # Within attempts
        # ---------------
        # Check: Documents grounded in generation
        doc_feedback = "Include the following content in your answer:"
        bad_doc_count = 0
        for doc in documents:
            exit = False
            limit = 3
            # This is necessary as LLM does not give fully consistent results (might miss out on certain fields in json)
            while not exit and limit > 0:
                try:
                    reply = self.custom_llm.hallucination_grader(
                        generated_answer=generation,
                        documents=doc.page_content
                    )
                    exit = True
                    grade = reply["score"]
                    reason = reply["reason"]
                except:
                    limit -= 1

            if limit == 0:
                print("Error: Could not get hallucination score.")
                grade = "no"
                reason = "Error: Could not get hallucination score."	

            if grade == "no":
                # Answer is not based on this document.
                doc_feedback += "\n\n" + reason
                bad_doc_count += 1

            # else pass

        print(f"Feedback: Answer deviated from {bad_doc_count} documents.")

        # Check: Answer useful
        ans_useful = True
        ans_check = self.custom_llm.answer_grader(generated_answer=generation, qn=question, chat_history=chat_history)
        # Try again till we get a score
        count = 0
        while "score" not in ans_check and count < 3:
            ans_check = self.custom_llm.answer_grader(generated_answer=generation, qn=question, chat_history=chat_history)
            count += 1
        
        # If still don't have
        if "score" not in ans_check:
            print("Using default score of \"no\"")
            ans_check = {"score": "no"}
        if ans_check["score"] == "no":
            print("Feedback: Answer not useful in answering the question.")
            answer_feedback = "Improve your answer by addressing the following points:"
            answer_feedback += "\n\n" + self.custom_llm.generate_ans_feedback(generated_answer=generation, qn=question, chat_history=chat_history,)
            ans_useful = False
        else:
            print("Feedback: Answer useful in answering the question.")

        # Collate feedback
        feedback = doc_feedback if bad_doc_count > 0 else ""
        feedback += "\n\n" + answer_feedback if not ans_useful else ""

        attempts += 1
        if attempts > 1:
            # If max attempts reached, some documents still not useful but answer is useful
            # Return generation
            if ans_useful:
                return {"generation": generation, "verdict": "completed", "documents": documents}
            else:
                # This is if the rag approach fails to generate a meaningful response
                print("Max attempts reached.")
                # Max number of retries reached
                # Suggests the loop does not produce meaningful results
                # Return generation 
                generation = "_I'm not confident in this answer, but here's my attempt._\n\n" + generation
                return {"generation": generation, "verdict": "completed", "documents": documents}


        # If feedback is empty, we can end the conversation by returning "completed"
        if feedback == "":
            return {"generation": generation, "verdict": "completed", "documents": documents}
        else:
            # Return feedback and increase attempts (and return it)
            cprint("---" * 20, "red")
            cprint(f"Final feedback: {feedback}", "red")
            cprint("---" * 20, "red")
            cprint(f"Number of attempts: {attempts}", "red")
            # limit feedback due to context size restrictions
            feedback = feedback[:4000]
            return {"feedback": feedback, "attempts": attempts, "verdict": "generate", "prev_ans": generation, "documents": documents}
        
    def critic_rerouter(self, state):
        return state["verdict"]