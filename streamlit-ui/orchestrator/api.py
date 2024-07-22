from flask import Flask, request, json, send_file, jsonify
from pprint import pprint
from workflow import buildChatApp
import traceback
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import Ollama
from app_config import config
from termcolor import cprint
import os

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
    # returns empty list by default
    convo_hist = data.get('convo_hist', [])
    # print()
    # cprint("CONVERSATION HISTORY", 'green')
    # cprint("---" * 10, "green")
    # cprint(convo_hist, "green")
    # print()

    def process_convo_hist(convo_hist):
        processed_hist = []
        # convert to AIMessage and HumanMessage objects
        for msg in convo_hist:
            role, content = msg["role"], msg["content"]
            if role == "user":
                processed_hist.append(HumanMessage(content=content))
            else:
                processed_hist.append(AIMessage(content=content))

        return processed_hist

    chat_history[tenant_id] = process_convo_hist(convo_hist)

    inputs = {"question": prompt, "chat_history": chat_history[tenant_id]}
    print("Building ChatApp")
    chat_app = buildChatApp(tenant_id, chat_mode)

    try:
        for output in chat_app.stream(inputs):
            for key, value in output.items():
                print()
                cprint(f"Finished running: {key}", "yellow")
                print()

        response = value["generation"]
        cprint("---" * 20, "green")
        cprint(response, "green")
        cprint("---" * 20, "green")

        # Store chat history
        chat_history[tenant_id].append(HumanMessage(content=prompt))
        chat_history[tenant_id].append(AIMessage(content=response))

        if "documents" in value:
            documents = [dict(doc.metadata, page_content=doc.page_content) for doc in value["documents"]]  # convert metadata to dict
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

        cprint("Server has responded.", "green")

        # json string
        return json.dumps(data)

    except Exception as e:
        cprint(f"Error location: {traceback.format_exc()}", "red")
        cprint(f"Error: {e}", "red")
    
    return f"I'm sorry, I'm unable to generate a response at the moment. Please try again later."

@app.route('/topics', methods=['GET'])
def topics_endpoint():
    try:
        # Path to your saved topic model file
        file_path = "topic_cache"

        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010)
