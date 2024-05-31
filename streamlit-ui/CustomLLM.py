from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

class CustomLLM:
    def __init__(self):
        self.json_parser = JsonOutputParser()
        self.json_llm = ChatOllama(model="llama3", temperature=0, format="json")
        self.llm = ChatOllama(model="llama3", temperature=0)


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

        routing_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at deciding whether a question
            is worth answering or not. Questions that are not worth answering are the ones that are rude or asks you to perform
            image generation tasks. Questions that asks you about your knowledge base are the ones that you SHOULD answer, especially
            those summary or analysis-based-on-knowledge tasks.
            Give a binary choice of 'give up' if question is not worth answering, or 'vectorstore' if the question is worth answering.
            Return a JSON with a single key 'datasource' and 
            no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
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
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
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
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use ten sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {question} 
            Context: {context} 
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question", "document"],
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
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "documents"],
        )

        hallucination_grading_pipeline = hallucination_prompt | self.json_llm | JsonOutputParser()

        return hallucination_grading_pipeline.invoke(
                {"generation": generated_answer, "documents": documents}
            )


    def answer_grader(self, generated_answer: str, qn: str):
        """
        Use LLM to grade whether an answer is useful for a question.

        Args:
            generated_answer (str): The generated answer
            qn (str): The user question

        Returns:
            str: The grade of the answer (either yes or no)
        """
        ans_grader_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "question"],
        )

        answer_grader = ans_grader_prompt | self.json_llm | JsonOutputParser()

        return answer_grader.invoke({"question": qn, "generation": generated_answer})
