from time import sleep
from pprint import pprint
from workflow import buildChatApp
import traceback
import time

def entry(prompt, st, tenant_id, chat_mode="All-Purpose"):
    """
    Generates the response from the workflow.

    Args:
        prompt (str): The prompt to generate a response for
        st (streamlit): The streamlit object
        tenant_id (str): The tenant ID
        chat_mode (str): The chat mode ie All-Purpose, RAG, Chatbot. All-Purpose by default.

    Returns:
        str: The response from the workflow
    """

    inputs = {"question": prompt}
    # this is the compiled workflow,
    # acting as a black box
    chat_app = buildChatApp(tenant_id, chat_mode)

    # run the workflow
    ## there might be errors (llm not producing a json when required)
    tries = 0
    max_tries = 3
    while tries < max_tries:
        try:
            start_time = time.time()
            for output in chat_app.stream(inputs):
                for key, value in output.items():
                    print(f"Finished running: {key}:")
                    pprint(value)
                    print()

                # If response takes longer than 60s, return a timeout error
                if time.time() - start_time > 60:
                    print(TimeoutError("Process took too long to complete"))
                    return "I'm sorry, I'm unable to generate a response at the moment. Please try again later or change your prompt."
            
            return value["generation"]
        
        except Exception as e:
            tries += 1
            st.write(f"Error: {e}")
            st.write(f"Error location: {traceback.format_exc()}")
            st.write(f"Retrying... {tries}/{max_tries}")
            print(f"Error location: {traceback.format_exc()}")
            print(f"Error: {e}")
            continue
    
    return "I'm sorry, I'm unable to generate a response at the moment. Please try again later."

