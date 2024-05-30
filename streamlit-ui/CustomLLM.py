from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class CustomLLM:
    def __init__(self):
        # TODO: Move this to another file
        self.config = {"ollama_base_url": "http://localhost:11434",
                        "llm_name": "llama3",
                        }

        # load the llm
        self.llm = ChatOllama(
                    temperature=0,
                    base_url=self.config["ollama_base_url"],
                    model=self.config["llm_name"],
                    streaming=True,
                    # seed=2,
                    top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
                    top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
                    num_ctx=3072,  # Sets the size of the context window used to generate the next token.
                )

    def initial_router(self, prompt: str) -> str:
        """
        LLM evaluates the question to determine if we should look it up in the vectorstore or the web
        based on a predefined set of topics as mentioned in the prompt.
        
        Args:
            prompt (str): The user prompt to evaluate

        Returns:
            str: Either the vectorstore or give up
        """

        routing_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at deciding whether a question
            is worth answering or not. Questions that are not worth answering are primarily those that requires you to perform
            non-text generation tasks such as generating an image or video. You do not need to be stringent with the keywords 
            in the question. If questions are not worth answering, give up. Otherwise, look into the vectorstore.
            Give a binary choice of 'give up' or 'vectorstore' based on the question. Return the JSON with a single key 'datasource' and 
            no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question"],
        )

        router = routing_prompt | self.llm | JsonOutputParser()

        return router.invoke({"question": prompt})

