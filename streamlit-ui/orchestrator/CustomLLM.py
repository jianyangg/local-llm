from langchain_community.llms import Ollama
# from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from app_config import config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from termcolor import cprint

# TODO: Add additional variable into graph state to keep feedback on the responses from grader, should there be retries.
class CustomLLM:
    def __init__(self, tenant_id: str):
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()
        self.json_llm = Ollama(model=config["llm_name"], temperature=0, format="json", base_url=config["ollama_base_url"], verbose=False)
        self.llm = Ollama(model=config["llm_name"], temperature=0, base_url=config["ollama_base_url"], verbose=False)
        self.tenant_id = tenant_id
        chat_hist_summariser_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a summarizer tasked with collating what ONE party has said.
            You are to provide a brief overview of the content.
            The summary should be concise and capture the key points of the content.
            Return the summary with no preamble or explanation.
            If there's nothing to summarise, return an empty string.
            YOU HAVE NO PERSONALITY and YOURE JUST A TOOL. DO NOT REPLY TO THE USER, JUST DO YOUR JOB.
            <|start_header_id|>user<|end_header_id|>
            Content: {chat_history}
            DO NOT BE VERBOSE. BE VERY CONCISE.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["chat_history"],
        )
        self.chat_hist_summariser = chat_hist_summariser_prompt | self.llm | self.str_parser
    
    def get_summarised_prev_ans(self, prev_ans: str):
        summarised_prev_ans = self.llm.invoke(f"Summarise and be concise: {prev_ans}")
        return summarised_prev_ans

    def get_summarised_chat_hist(self, chat_history: list):
        # Separetely summarise the chat history for human and AI messages
        human_convo_hist_list = [msg.content for idx, msg in enumerate(chat_history) if idx % 2 == 0]
        human_convo_hist = " ".join(human_convo_hist_list)
        ai_convo_hist_list = [msg.content for idx, msg in enumerate(chat_history) if idx % 2 == 1]
        ai_convo_hist = " ".join(ai_convo_hist_list)
        summarised_human_hist = self.chat_hist_summariser.invoke({"chat_history": human_convo_hist})
        summarised_ai_hist = self.chat_hist_summariser.invoke({"chat_history": ai_convo_hist})
        collated_summaries = [HumanMessage(content=summarised_human_hist), AIMessage(content=summarised_ai_hist)]
        return collated_summaries

    def initial_router(self, prompt: str, chat_history: list):
        """
        LLM evaluates the question to determine if we should look it up in the vectorstore or the web
        based on a predefined set of topics as mentioned in the prompt.
        
        Args:
            prompt (str): The user prompt to evaluate

        Returns:
            str: Either the vectorstore or give up
        """

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
    
    def retrieval_grader(self, qn: str, doc_content: str):
        """
        Use LLM to grade the relevance of a document to a user question.

        Args:
            qn (str): The user question
            doc_content (str): The document content

        Returns:
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

            # these refer to the items in {} in the template above
            input_variables=["question", "document"],
        )

        # uses a pipeline to put the prompt through the LLM and then parse the output
        doc_grading_pipeline = doc_grader_prompt | self.json_llm | self.json_parser

        return doc_grading_pipeline.invoke(
            {"question": qn, "document": doc_content}
        )
    
    def answer_generator(self, context: str, qn: str, chat_history: list, feedback: str, prev_ans = ""):
        """
        Use LLM to generate an answer to a user question based on a context.
        
        Args:
            context (str): The context to generate the answer from
            qn (str): The user question

        Returns:
            str: The generated answer
        """

        answerer_prompt_no_feedback = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    <|begin_of_text|>
                    <|start_header_id|>system<|end_header_id|>
                    You are called a DSTA Chatbot called Jarvis.
                    You are to be helpful, friendly, and professional.
                    You are given a query and a set of documents.
                    Answer the query based on the documents and the chat history provided below.
                    <|start_header_id|>user<|end_header_id|>
                    ONLY REFERENCE THE CHAT HISTORY IF IT HELPS YOU ANSWER THE QUESTION.
                    Chat History:
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
                    Answer this question: {query}
                    Documents to reference: {documents}
                    Your task is to provide a detailed and well-structured answer based on the documents provided.
                    The documents have all been pre-processed and are determined by your overlords to be relevant to the query -- do not second-guess them.
                    Please ensure that your answer is clear, concise, and divided into the following sections:
                    1. **Introduction**: Briefly summarize the query and the context.
                    2. **Key Information from Documents**: Highlight the most relevant information from the documents that directly addresses the query.
                    3. **Detailed Answer**: Provide a thorough and detailed answer to the query, integrating information from the documents.
                    4. **Conclusion**: Summarize the key points and provide any additional insights or recommendations if relevant.
                    Remember to keep your answers concise and structured.
                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    """
                )
            ]
        )

        answerer_prompt_w_feedback = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    <|begin_of_text|>
                    <|start_header_id|>system<|end_header_id|>
                    You are a highly knowledgeable and structured Retrieval QA model. You are given a query and a set of documents.
                    <|start_header_id|>user<|end_header_id|>
                    ONLY REFERENCE THE CHAT HISTORY IF IT HELPS YOU ANSWER THE QUESTION.
                    Chat History:
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
                    Feedback: {feedback}
                    Your task is to EDIT THE FOLLOWING ANSWER BASED ON THE PROVIDED FEEDBACK
                    MINIMISE EDITS TO THE ANSWER. SLIGHT MODIFICATIONS BASED ON THE FEEDBACK ONLY.
                    Answer to be improved: {prev_ans}
                    Please ensure that your answer is clear, concise, and divided into the following sections:
                    1. **Introduction**: Briefly summarize the query and the context.
                    2. **Key Information from Documents**: Highlight the most relevant information from the documents that directly addresses the query.
                    3. **Detailed Answer**: Provide a thorough and detailed answer to the query, integrating information from the documents.
                    4. **Conclusion**: Summarize the key points and provide any additional insights or recommendations if relevant.
                    Remember to keep your answers concise and structured.
                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    """
                )
            ]
        )

        answer_pipeline = (answerer_prompt_no_feedback if feedback == "" else answerer_prompt_w_feedback) | self.llm | StrOutputParser()
        cprint("Running history summariser", "yellow")
        summarised_chat_history = self.get_summarised_chat_hist(chat_history)
        cprint(f"Summarised chat history: {summarised_chat_history}", "yellow")
        generated_answer = answer_pipeline.invoke({"query": qn, "documents": context, "chat_history": summarised_chat_history}) if feedback == "" else answer_pipeline.invoke({"query": qn, "documents": context, "chat_history": summarised_chat_history, "feedback": feedback, "prev_ans": prev_ans})
        print("---" * 5)
        print("Draft answer:")
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

        hallucination_prompt = PromptTemplate(
            # Template for llama3
            template="""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are a grader assessing whether the answer has referenced the given factual information.
            Give a binary 'yes' or 'no' score to indicate 
            whether a piece of factual information is found in the answer. 
            Provide the binary score as a JSON with a 
            one key 'score' and just one more key "reason" with the content to add.

            Format: "score": "yes" or "score": "no", "reason": REPLACE_W_CONTENT_TO_ADD

            Content to add should be DIRECT instead of passive, such as "Add abc" instead of "It's missing abc".
            Give the SPECIFIC content to add, instead of a general idea of it. In other words, give the answer instead of general guidance.

            Give a "yes" if ANY part of the answer references the factual document chunk.
            Give a "no" otherwise, but justify your answer with a VERY BRIEF explanation.
            The "reason" should have DIRECT ACTIONABLE FEEDBACK.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the reference document chunk:
            \n ------- \n
            {document} 
            \n ------- \n
            Here is the answer: {generation}
            Remember to format your answer as either "score": "yes" or "score": "no"
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",

            input_variables=["generation", "document"],
        )

        hallucination_grading_pipeline = hallucination_prompt | self.json_llm | JsonOutputParser()

        return hallucination_grading_pipeline.invoke(
                {"generation": generated_answer, "document": documents}
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
                    useful to resolve a question. Provide the binary score as a JSON with a key 'score' and another key "reason" on how to improve the answer.
                    Format: "score": "yes" or "score": "no", "reason": REPLACE_W_DIRECT_ACTIONABLE_FEEDBACK
                    Do not provide any preamble or explanation, especially for the reason.
                    The reason should have DIRECT ACTIONABLE FEEDBACK. For example, instead of saying "Not useful", say "Add more details on xyz".
                    Keep the reason CONCISE.
                    <|start_header_id|>user<|end_header_id|>
                    Chat History:
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
                    Here is the answer:
                    \n ------- \n
                    {generation} 
                    \n ------- \n
                    Here is the question: {question}
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    """
                )
            ]
        )

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
                    <|start_header_id|>user<|end_header_id|>
                    Chat History:
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
                    Question: {question}
                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
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
                    <|begin_of_text|>
                    <|start_header_id|>system<|end_header_id|>
                    You are a rephraser. 
                    Your task is to read through the following messages and rephrase the user's question to add more context, making it more specific and meaningful for document retrieval. 
                    The message immediately preceding the question is likely to be the most relevant for rephrasing.
                    It's optional to use the chat history to identify key context and keywords that can make the rephrased question more specific.
                    Do not respond to the user; only rephrase the question.
                    Do not rephrase text within double quotes.
                    <|start_header_id|>user<|end_header_id|>
                    Chat History:
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
                    Question to rephrase: {question}
                    Provide your rephrased question directly without any preamble or additional response. 
                    If you do not think the question needs rephrasing, respond with the original question.
                    DO NOT include any additional information or context in your response.
                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    """
                )
            ]
        )

        rephraser_chain = rephraser_prompt | self.llm | StrOutputParser()

        rephrased_prompt = rephraser_chain.invoke({"question": qn, "chat_history": chat_history})
        cprint(f"Original question: {qn}", "red")
        cprint(f"Rephrased question for retrieval: {rephrased_prompt}", "green")
        return rephrased_prompt
    
    def generate_ans_feedback(self, generated_answer: str, qn: str, chat_history: list):
        """
        Use LLM to grade whether an answer is useful for a question.
        """
        summarised_chat_history = self.get_summarised_chat_hist(chat_history)

        feedback_prompt = PromptTemplate(
            template="""
                <begin_of_text>
                <|start_header_id|>system<|end_header_id|>
                The answer is deemed to be not useful in answering the question.
                Review the answer in light of the question and the chat history given below.
                Provide feedback on how to improve the answer and suggest improvements in a CLEAR and CONCISE manner.
                Limit your response to at most 2 sentences.
                Exclude any preamble or explanation.
                <|start_header_id|>user<|end_header_id|>
                Chat History: {chat_history}
                Question: {question}
                Answer: {generation}
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["question", "generation", "chat_history"],
        )

        feedback_pipeline = feedback_prompt | self.llm | StrOutputParser()

        return feedback_pipeline.invoke({"question": qn, "generation": generated_answer, "chat_history": summarised_chat_history})