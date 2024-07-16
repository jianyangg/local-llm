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
            You are a summarizer focused on extracting and condensing the statements made by a single designated party within a chat conversation.

            Your goal is to:

            1. Identify all messages belonging to the target party.
            2. Extract the core ideas and information from those messages.
            3. Synthesize a concise summary that reflects the essence of their communication.
            4. If the target party has not spoken or if there is no chat history to analyze, return an empty string.

            Crucial Instructions:

            * Your response should ONLY be the summary; do not include any introductory phrases or explanations.
            * Prioritize brevity and focus on the most salient points.
            * Avoid adding your own interpretations or opinions.

            <|start_header_id|>user<|end_header_id|>
            Target Party: [Name or identifier of the person you want to summarize]
            Content: {chat_history} 
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["chat_history"],
        )
        self.chat_hist_summariser = chat_hist_summariser_prompt | self.llm | self.str_parser

    def get_summarised_chat_hist(self, chat_history: list):
        # Separetely summarise the chat history for human and AI messages
        human_convo_hist_list = [msg.content for idx, msg in enumerate(chat_history) if idx % 2 == 0]
        print("Length of convo hist (human):", len(human_convo_hist_list))
        human_convo_hist = " ".join(human_convo_hist_list)
        ai_convo_hist_list = [msg.content for idx, msg in enumerate(chat_history) if idx % 2 == 1]
        print("Length of convo hist (AI):", len(ai_convo_hist_list))
        if len(ai_convo_hist_list) == 0:
            # The first run don't need chat history
            return [HumanMessage(content=""), AIMessage(content="")]
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
            Generic questions that can be answered without data should be answered directly through generation so 'generate'.
            If questions refer you to a knowledge base or a data source or might be supplemented with documents in knowledge base,
            you should route the question to the 'vectorstore'.
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
                    Query: {query}
                    Feedback: {feedback}
                    Documents to reference: {documents}
                    Your task is to EDIT THE FOLLOWING ANSWER BASED ON THE PROVIDED FEEDBACK
                    MINIMISE EDITS TO THE ANSWER. SLIGHT MODIFICATIONS BASED ON THE FEEDBACK ONLY.
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
        generated_answer = answer_pipeline.invoke({"query": qn, "documents": context, "chat_history": summarised_chat_history}) if feedback == "" else answer_pipeline.invoke({"query": qn, "documents": context, "chat_history": summarised_chat_history, "feedback": feedback})
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
                    You are the DSTA chatbot named Jarvis. Your primary goal is to assist users effectively.

                    Guidelines:

                    Be helpful, friendly, and professional in your responses.
                    Thoroughly analyze the user's question and chat history.
                    If the question is unclear or requires more information, ask clarifying questions.
                    If you can confidently answer the question, provide a concise, accurate response.
                    Avoid speculation or assumptions.
                    Maintain a professional tone throughout the conversation. <|start_header_id|>user<|end_header_id|> 
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
                    You are an expert at rephrasing questions for document retrieval.
                    You are to make VERY SLIGHT edits to the question to make it more suitable for retrieval.
                    DO NOT change any nouns, entities, organisations, persons, topics or the core meaning of the question.
                    Only ADD words from the chat history to provide additional context, BUT ONLY IF CONTEXT IS NEEDED.
                    <|start_header_id|>user<|end_header_id|>
                    Chat History:
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
                    Question to improve: {question}
                    REMINDER:
                    You are to make VERY SLIGHT edits to the question to make it more suitable for retrieval.
                    DO NOT change any nouns, entities, organisations, persons, topics or the core meaning of the question.
                    Only ADD words from the chat history to provide additional context, BUT ONLY IF CONTEXT IS NEEDED.
                    DO NOT include any preamble or explanation.
                    IMPORTANT! DO NOT ADD ITEMS FROM CHAT HISTORY IF IT IS NOT NEEDED. 
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
                Evaluate the answer's relevance to the question and chat history.
                Provide clear, concise feedback (max. 2 sentences) on how to improve the answer.
                Focus on specific changes and avoid general comments.
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