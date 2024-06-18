# from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

class CustomLLM:
    def __init__(self, tenant_id: str):
        self.json_parser = JsonOutputParser()

        # For testing with llama3
        self.json_llm = Ollama(model="llama3", temperature=0, format="json", base_url="http://localhost:11434")
        # TODO: Add additional variable into graph state to keep feedback on the responses from grader, should there be retries.
        self.llm = Ollama(model="llama3", temperature=0.2, base_url="http://localhost:11434")
        # TODO: Use this tenant_id when querying documents
        self.tenant_id = tenant_id

        # ## For testing with phi2
        # self.json_llm = Ollama(model="phi", temperature=0, format="json", base_url="http://localhost:11435")
        # self.llm = Ollama(model="phi", temperature=0.1, base_url="http://localhost:11435")

    def initial_router(self, prompt: str):
        """
        LLM evaluates the question to determine if we should look it up in the vectorstore or the web
        based on a predefined set of topics as mentioned in the prompt.

        # TODO: This can be used as the filtering mechanism
        
        Args:
            prompt (str): The user prompt to evaluate

        Returns:
            # TODO: this might not be str
            str: Either the vectorstore or give up
        """

        # TODO: Include a chat_history prompt to keep track of the conversation: Chat History: ```{history}```"""
        # TODO: We can store this in the GraphState as well.
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

            ## Template for phi2
            # template="""You are an expert at deciding whether a question
            # requires the retrieval of information from a vectorstore before generating a response or to go straight to generating a response.
            # Generic questions that can be answered without data are not worth
            # routing to the vectorstore and should be answered directly through generation.
            # If questions refer you to a knowledge base or a data source, you should route the question to the 'vectorstore'. Otherwise, you should
            # reply 'generate' to answer the question directly.
            # If the question DOES NOT hint at the use of your knowledge base, you should route the question to 'generate'.
            # Give a binary choice of 'generate' if question requires NO reference to the datasource, and 'vectorstore' if it does.
            # Return a JSON with a single key 'datasource' and 
            # no preamble or explanation. Question to route: {question}""",
            input_variables=["question"],
        )

        routing_pipeline = routing_prompt | self.json_llm | self.json_parser

        return routing_pipeline.invoke({"question": prompt})
    

    def retrieval_grader(self, qn: str, doc_content: str):
        """
        Use LLM to grade the relevance of a document to a user question.

        Args:
            qn (str): The user question
            doc_content (str): The document content

        Returns:
            # TODO: this might not be str
            str: The grade of the document
        """
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

            ## Template for phi2
            # template="""You are a grader assessing relevance 
            # of a retrieved document to a user question. If the document contains keywords related to the user question, 
            # grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            # Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            # Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
            # <|eot_id|><|start_header_id|>user<|end_header_id|>
            # Here is the retrieved document: \n\n {document} \n\n
            # Here is the user question: {question} \n
            # """,

            # these refer to the items in {} in the template above
            input_variables=["question", "document"],
        )
        # uses a pipeline to put the prompt through the LLM and then parse the output
        doc_grading_pipeline = doc_grader_prompt | self.json_llm | JsonOutputParser()
        
        return doc_grading_pipeline.invoke(
                {"question": qn, "document": doc_content}
            )
    

    def answer_generator(self, context: str, qn: str):
        """
        Use LLM to generate an answer to a user question based on a context.
        
        Args:
            context (str): The context to generate the answer from
            qn (str): The user question

        Returns:
            # TODO: this might not be str
            str: The generated answer
        """
        answer_prompt = PromptTemplate(
            # """
            # Template for llama3
            # """
            template="""<|begin_of_text|>
                        <|start_header_id|>system<|end_header_id|>
                        You are Jarvis, an AI assistant designed to answer questions using the provided context.
                        Your responses must be grounded in the context given. 

                        Follow these instructions carefully:

                        1. **Analyze the Context:** Thoroughly examine the context to identify relevant information related to the question.
                        2. **Directly Answer the Question:** Craft your response based on the information found in the context. Prioritize accuracy and relevance.
                        3. **Cite Your Sources:** Clearly indicate which documents from the context support your answer.
                        4. **Admit Lack of Knowledge:** If the context does not contain enough information to answer the question definitively, state that you don't have the answer rather than guessing or making assumptions.
                        5. **Optional: Suggest Further Exploration:** If appropriate, based on the context, you may offer additional insights or suggest where the user could find more information.

                        Formatting:

                        * Use markdown for clear structure (headings, lists, etc.).
                        * Denote new lines with \n.
                        * Maintain a professional and concise tone. 

                        Format:
                        1. Title
                        2. Brief Introduction to answer
                        3. Detailed Answer with supporting evidence
                        4. Sources, specific file names and pages of the documents used.

                        Important:

                        * Do not include information that is not present in the context.
                        * Avoid speculation or personal opinions.
                        * Focus on providing a factual and informative response based on the evidence available.

                        <|eot_id|><|start_header_id|>user<|end_header_id|>

                        Question: {question} 
                        Context: {context}

                        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            
            # """
            # Template for Microsoft Phi-2
            # """
            # template="""
            #             Jarvis, you are an AI assistant designed to answer questions using the provided context.
            #             Your responses must be grounded in the context given. 

            #             Follow these instructions carefully:

            #             1. **Analyze the Context:** Thoroughly examine the context to identify relevant information related to the question.
            #             2. **Directly Answer the Question:** Craft your response based on the information found in the context. Prioritize accuracy and relevance.
            #             3. **Cite Your Sources:** Clearly indicate which documents from the context support your answer.
            #             4. **Admit Lack of Knowledge:** If the context does not contain enough information to answer the question definitively, state that you don't have the answer rather than guessing or making assumptions.
            #             5. **Optional: Suggest Further Exploration:** If appropriate, based on the context, you may offer additional insights or suggest where the user could find more information.

            #             Formatting:

            #             * Use markdown for clear structure (headings, lists, etc.).
            #             * Denote new lines with \n.
            #             * Maintain a professional and concise tone. 

            #             Format:
            #             1. Title
            #             2. Brief Introduction to answer
            #             3. Detailed Answer with supporting evidence
            #             4. Sources, specific file names and pages of the documents used.

            #             Important:

            #             * Do not include information that is not present in the context.
            #             * Avoid speculation or personal opinions.
            #             * Focus on providing a factual and informative response based on the evidence available.

            #             Keep your answer to 3 lines.

            #             Question: {question} 
            #             Context: {context}
            #             Answer: 
            #             """,
            input_variables=["question", "context"],
        )

        answer_pipeline = answer_prompt | self.llm | StrOutputParser()

        generated_answer = answer_pipeline.invoke({"question": qn, "context": context})
        print("Answer attempt (not final):", generated_answer)
        return generated_answer
    

    def hallucination_grader(self, generated_answer: str, documents: str):
        """
        Use LLM to grade whether an answer is grounded in the filtered document list.

        Args:
            generated_answer (str): The generated answer
            documents (str): The filtered documents

        Returns:
            # TODO: this might not be str
            str: The grade of the answer
        """
        hallucination_prompt = PromptTemplate(
            # Template for llama3
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",

            ## Template for phi2
            # template="""You are a grader assessing whether 
            # an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
            # whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            # single key 'score' and no preamble or explanation. 
            # Here are the facts:
            # \n ------- \n
            # {documents} 
            # \n ------- \n
            # Here is the answer: {generation} """,
            input_variables=["generation", "documents"],
        )

        hallucination_grading_pipeline = hallucination_prompt | self.json_llm | JsonOutputParser()

        return hallucination_grading_pipeline.invoke(
                {"generation": generated_answer, "documents": documents}
            )


    def answer_grader(self, generated_answer: str, qn: str):
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
        ans_grader_prompt = PromptTemplate(
            ## Template for llama3
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",

            ## Template for phi2
            # template="""You are a grader assessing whether an 
            # answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            # useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            # Here is the answer:
            # \n ------- \n
            # {generation} 
            # \n ------- \n
            # Here is the question: {question}""",
            input_variables=["generation", "question"],
        )

        answer_grader = ans_grader_prompt | self.json_llm | JsonOutputParser()

        return answer_grader.invoke({"question": qn, "generation": generated_answer})
