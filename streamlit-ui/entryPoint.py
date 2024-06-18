from time import sleep
from pprint import pprint
from workflow import buildChatApp
import traceback
import time

def entry(prompt, st, tenant_id):
    """
    The main entry point of the workflow.

    Args:
        streamlit_obj (streamlit): The streamlit object

    Returns:
        str: The response from the workflow

    """

    # this makes the state global so that we can call write
    # from another function w/o passing the state

    response = generate(prompt, st, tenant_id)
    return response


def generate(prompt, st, tenant_id):
    """
    Generates the response from the workflow.

    Args:
        prompt (str): The prompt to generate the response from

    Returns:
        str: The response from the workflow
    """

    inputs = {"question": prompt}
    # this is the compiled workflow,
    # acting as a black box
    chat_app = buildChatApp(tenant_id)

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

                # return the response to animate
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

