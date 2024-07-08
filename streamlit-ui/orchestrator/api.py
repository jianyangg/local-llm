from flask import Flask, request, json
from pprint import pprint
from workflow import buildChatApp
import traceback
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import Ollama
from app_config import config

# This is the only file that receives both inputs and outputs.
# So it's convenient to store the chat history here.

app = Flask(__name__)
# stores chat history, similar to the logic in jarvis.py
# dictionary to store for multiple users.
chat_history = {}

@app.route('/entry', methods=['POST'])
def entry_endpoint():
    data = request.get_json()
    prompt = data.get('prompt')
    tenant_id = data.get('tenant_id')
    chat_mode = data.get('chat_mode', 'Semantic Search w/o Agents')
    generate_title = data.get('generate_title', False)

    # Initialise chat history
    if tenant_id not in chat_history:
        chat_history[tenant_id] = []


    print("Chat history for tenant_id:", chat_history[tenant_id])
    # TODO: Fixed number of 6. Could be made dynamic.
    inputs = {"question": prompt, "chat_history": chat_history[tenant_id][-6:]}
    print("Building ChatApp")
    chat_app = buildChatApp(tenant_id, chat_mode)

    try:
        for output in chat_app.stream(inputs):
            for key, value in output.items():
                print(f"Finished running: {key}:")
                pprint(value)
                print()

        print(value)
        response = value["generation"]

        # Store chat history
        chat_history[tenant_id].append(HumanMessage(content=prompt))
        chat_history[tenant_id].append(AIMessage(content=response))

        if "documents" in value:
            documents = [dict(doc.metadata, page_content=doc.page_content) for doc in value["documents"]]  # convert metadata to dict
            print("Server done.")
            print(f"Response: {response}")  
            print(f"Documents: {documents}")
            data = {"response": response, "documents": documents}
        else:
            data = {"response": response}

        # generate title if required
        if generate_title:
            llm = Ollama(model=config["llm_name"], temperature=0, base_url=config["ollama_base_url"])
            # generate title prompt
            prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Generate a 4 word title to categorise the following question and answer pair:
            <|start_header_id|>user<|end_header_id|>
            Question: {prompt}
            Answer: {response}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            title = llm.invoke(prompt)
            print(f"Generated Title: {title}")
            data["title"] = title

        # json string
        return json.dumps(data)

    except Exception as e:
        print(f"Error location: {traceback.format_exc()}")
        print(f"Error: {e}")
    
    return f"I'm sorry, I'm unable to generate a response at the moment. Please try again later."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010)
