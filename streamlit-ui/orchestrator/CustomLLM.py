from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from app_config import config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# TODO: Add additional variable into graph state to keep feedback on the responses from grader, should there be retries.
# TODO: Include a chat_history prompt to keep track of the conversation: Chat History: ```{history}```"""
# TODO: We can store this in the GraphState as well.
class CustomLLM:
    def __init__(self, tenant_id: str):
        self.json_parser = JsonOutputParser()
        self.json_llm = Ollama(model=config["llm_name"], temperature=0, format="json", base_url=config["ollama_base_url"])
        self.llm = Ollama(model=config["llm_name"], temperature=0, base_url=config["ollama_base_url"])
        self.tenant_id = tenant_id

    def initial_router(self, prompt: str, chat_history: list):
        """
        LLM evaluates the question to determine if we should look it up in the vectorstore or the web
        based on a predefined set of topics as mentioned in the prompt.
        
        Args:
            prompt (str): The user prompt to evaluate

        Returns:
            str: Either the vectorstore or give up
        """

        # routing_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             """
        #             <|begin_of_text|>
        #             <|start_header_id|>system<|end_header_id|>
        #             You are an expert at deciding whether a question
        #             requires the retrieval of information from a vectorstore before generating a response or to go straight to generating a response.
        #             Generic questions that can be answered without data are not worth
        #             routing to the vectorstore and should be answered directly through generation.
        #             If questions refer you to a knowledge base or a data source, you should route the question to the 'vectorstore'. Otherwise, you should
        #             reply 'generate' to answer the question directly.
        #             If the question DOES NOT hint at the use of your knowledge base, you should route the question to 'generate'.
        #             Give a binary choice of 'generate' if question requires NO reference to the datasource, and 'vectorstore' if it does.
        #             Return a JSON with a single key 'datasource' and 
        #             no preamble or explanation.
        #             """
        #         ),
        #         MessagesPlaceholder(variable_name="chat_history"),
        #         (
        #             "human",
        #             """
        #             <|start_header_id|>user<|end_header_id|>
        #             Question to route: {question}
        #             <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        #             """
        #         )
        #     ]
        # )

        routing_prompt = PromptTemplate(
            # Template for llama3
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at deciding whether a question
            requires the retrieval of information from a vectorstore before generating a response or to go straight to generating a response.
            Generic questions that can be answered without data are not worth
            routing to the vectorstore and should be answered directly through generation.
            If questions refer you to a knowledge base or a data source, you should route the question to the 'vectorstore'. Otherwise, you should
            reply 'generate' to answer the question directly.
            If the question DOES NOT hint at the use of your knowledge base, you should route the question to 'generate'.
            Give a binary choice of 'generate' if question requires NO reference to the datasource, and 'vectorstore' if it does.
            Return a JSON with a single key 'datasource' and 
            no preamble or explanation.
            <|start_header_id|>user<|end_header_id|>
            Question to route: {question}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",

            input_variables=["question"],
        )

        routing_pipeline = routing_prompt | self.json_llm | self.json_parser

        return routing_pipeline.invoke({"question": prompt})
    
    def retrieval_grader(self, qn: str, doc_content: str, chat_history: list):
        """
        Use LLM to grade the relevance of a document to a user question.

        Args:
            qn (str): The user question
            doc_content (str): The document content

        Returns:
            str: The grade of the document
        """

        # doc_grader_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             """
        #             <|begin_of_text|>
        #             <|start_header_id|>system<|end_header_id|>
        #             You are a grader assessing relevance 
        #             of a retrieved document to a user question. If the document contains keywords related to the user question, 
        #             grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        #             Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        #             Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        #             """
        #         ),
        #         MessagesPlaceholder(variable_name="chat_history"),
        #         (
        #             "human",
        #             """
        #             <|eot_id|><|start_header_id|>user<|end_header_id|>
        #             Here is the retrieved document: \n\n {document} \n\n
        #             Here is the user question: {question} \n
        #             <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        #             """
        #         )
        #     ]
        # )

        doc_grader_prompt = PromptTemplate(
            # Template for llama3
            template="""<|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,

            # these refer to the items in {} in the template above
            input_variables=["question", "document"],
        )

        # uses a pipeline to put the prompt through the LLM and then parse the output
        doc_grading_pipeline = doc_grader_prompt | self.json_llm | self.json_parser
        
        # return doc_grading_pipeline.invoke(
        #         {"question": qn, "document": doc_content, "chat_history": chat_history}
        #     )
        return doc_grading_pipeline.invoke(
            {"question": qn, "document": doc_content}
        )
    
    def answer_generator(self, context: str, qn: str, chat_history: list):
        """
        Use LLM to generate an answer to a user question based on a context.
        
        Args:
            context (str): The context to generate the answer from
            qn (str): The user question

        Returns:
            str: The generated answer
        """

        answerer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    <|begin_of_text|>
                    <|start_header_id|>system<|end_header_id|>
                    You are a highly knowledgeable and structured Retrieval QA model. You are given a query and a set of documents.
                    Your task is to provide a detailed and well-structured answer based on the documents provided.
                    The documents have all been pre-processed and are determined by your overlords to be relevant to the query -- do not second-guess them.
                    Please ensure that your answer is clear, concise, and divided into the following sections:
                    1. **Introduction**: Briefly summarize the query and the context.
                    2. **Key Information from Documents**: Highlight the most relevant information from the documents that directly addresses the query.
                    3. **Detailed Answer**: Provide a thorough and detailed answer to the query, integrating information from the documents.
                    4. **Conclusion**: Summarize the key points and provide any additional insights or recommendations if relevant.
                    Remember to keep your answers concise and structured.
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                    Query: {query}
                    Documents: {documents}
                    <|eot_id|><|start_header_id|>assistant<|end_header_id>
                    """
                )
            ]
        )

        # answerer_prompt = PromptTemplate(
        #     template=""""
        #         <|begin_of_text|>
        #         <|start_header_id|>system<|end_header_id|>
        #         You are a highly knowledgeable and structured Retrieval QA model. You are given a query and a set of documents.
        #         Your task is to provide a detailed and well-structured answer based on the documents provided.
        #         The documents have all been pre-processed and are determined by your overlords to be relevant to the query -- do not second-guess them.
        #         Please ensure that your answer is clear, concise, and divided into the following sections:

        #         1. **Introduction**: Briefly summarize the query and the context.
        #         2. **Key Information from Documents**: Highlight the most relevant information from the documents that directly addresses the query.
        #         3. **Detailed Answer**: Provide a thorough and detailed answer to the query, integrating information from the documents.
        #         4. **Conclusion**: Summarize the key points and provide any additional insights or recommendations if relevant.

        #         Remember to keep your answers concise and structured.
        #         <|eot_id|><|start_header_id|>user<|end_header_id|>
        #         Query: {query}
        #         Documents: {documents}

        #         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        #         """,
        #     input_variables=["query", "documents"],
        # )

        answer_pipeline = answerer_prompt | self.llm | StrOutputParser()
        generated_answer = answer_pipeline.invoke({"query": qn, "documents": context, "chat_history": chat_history})
        print("---" * 5)
        print("Temp answer:")
        print(generated_answer)
        print("---" * 5)

        return generated_answer
    
    def hallucination_grader(self, generated_answer: str, documents: str):
        """
        Use LLM to grade whether an answer is grounded in the filtered document list.

        Args:
            generated_answer (str): The generated answer
            documents (str): The filtered documents

        Returns:
            str: The grade of the answer
        """

        # hallucination_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             """
        #             <|begin_of_text|>
        #             <|start_header_id|>system<|end_header_id|>
        #             You are a grader assessing whether 
        #             an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        #             whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        #             single key 'score' and no preamble or explanation.
        #             """
        #         ),
        #         MessagesPlaceholder(variable_name="chat_history"),
        #         (
        #             "human",
        #             """
        #             <|eot_id|><|start_header_id|>user<|end_header_id|>
        #             Here are the facts:
        #             \n ------- \n
        #             {documents} 
        #             \n ------- \n
        #             Here is the answer: {generation}
        #             <|eot_id|><|start_header_id|>assistant<|end_header_id>
        #             """
        #         )
        #     ]
        # )

        hallucination_prompt = PromptTemplate(
            # Template for llama3
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation.

            Format: "score": "yes" or "score": "no"
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}
            Remember to format your answer as either "score": "yes" or "score": "no"
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",

            input_variables=["generation", "documents"],
        )

        hallucination_grading_pipeline = hallucination_prompt | self.json_llm | JsonOutputParser()

        return hallucination_grading_pipeline.invoke(
                {"generation": generated_answer, "documents": documents}
            )

    def answer_grader(self, generated_answer: str, qn: str, chat_history: list):
        """
        Use LLM to grade whether an answer is useful for a question.
        TODO: Consider using this to check if the answering format is being adhered to.
        TODO: Else, include additonal guidelines in the GraphState.

        Args:
            generated_answer (str): The generated answer
            qn (str): The user question

        Returns:
            str: The grade of the answer (either yes or no)
        """

        ans_grader_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    <|begin_of_text|>
                    <|start_header_id|>system<|end_header_id|>
                    You are a grader assessing whether an 
                    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
                    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                    Here is the answer:
                    \n ------- \n
                    {generation} 
                    \n ------- \n
                    Here is the question: {question}
                    <|eot_id|><|start_header_id|>assistant<|end_header_id>
                    """
                )
            ]
        )

        # ans_grader_prompt = PromptTemplate(
        #     ## Template for llama3
        #     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        #     answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        #     useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        #     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        #     \n ------- \n
        #     {generation} 
        #     \n ------- \n
        #     Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",

        #     input_variables=["generation", "question"],
        # )

        answer_grader = ans_grader_prompt | self.json_llm | JsonOutputParser()

        return answer_grader.invoke({"question": qn, "generation": generated_answer, "chat_history": chat_history})

    def llm_only(self, qn: str, chat_history: list):
        """
        Use LLM to generate a response to a prompt.

        Args:
            qn (str): The prompt to generate a response for

        Returns:
            str: The generated response
        """

        system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    <|begin_of_text|>
                    <|start_header_id|>system<|end_header_id|>
                    You are called a DSTA Chatbot called Jarvis.
                    Answer the following question in a helpful, friendly, and PROFESSIONAL manner.
                    Ask follow up questions where you deem relevant.
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                    {question}
                    <|eot_id|><|start_header_id|>assistant<|end_header_id>
                    """
                )
            ]
        )

        chatbot_chain = system_prompt | self.llm | StrOutputParser()

        return chatbot_chain.invoke({"question": qn, "chat_history": chat_history})	
    
    def rephraser(self, qn: str, chat_history: list):
        """
        Rephrase the question in relation to the chat_history to improve the quality of the answer.

        Args:
            state (dict): The current graph state

        Returns:
            str: The rephrased question
        """
            
        rephraser_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a rephraser. 
                    Your task is to read through the following messages and rephrase the user's question to add more context, making it more specific and meaningful for document retrieval. 
                    The message immediately preceding the question is likely to be the most relevant for rephrasing.
                    Use the chat history to identify key context and keywords that can make the rephrased question more specific.
                    Do not respond to the user; only rephrase the question.
                    Do not rephrase text within double quotes.
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
                    user
                    Question to rephrase: {question}
                    Provide your rephrased question directly without any preamble or additional response. 
                    """
                )
            ]
        )

        rephraser_chain = rephraser_prompt | self.llm | StrOutputParser()

        rephrased_prompt = rephraser_chain.invoke({"question": qn, "chat_history": chat_history})
        print("Rephrased question:")
        print(rephrased_prompt)
        return rephrased_prompt