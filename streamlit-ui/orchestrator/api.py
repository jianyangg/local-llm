from flask import Flask, request
from pprint import pprint
from workflow import buildChatApp
import traceback
import time

app = Flask(__name__)

@app.route('/entry', methods=['POST'])
def entry_endpoint():
    data = request.get_json()
    prompt = data.get('prompt')
    tenant_id = data.get('tenant_id')
    chat_mode = data.get('chat_mode', 'All-Purpose')

    inputs = {"question": prompt}
    chat_app = buildChatApp(tenant_id, chat_mode)

    tries = 0
    max_tries = 3
    e = "" # error placeholder
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
            print(f"Error location: {traceback.format_exc()}")
            print(f"Error: {e}")
            continue
    
    return f"I'm sorry, I'm unable to generate a response at the moment. Please try again later. Error {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
