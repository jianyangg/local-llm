from flask import Flask, request
from pprint import pprint
from workflow import buildChatApp
import traceback
import time
from utils import delete_screenshots
import os

app = Flask(__name__)

@app.route('/entry', methods=['POST'])
def entry_endpoint():
    data = request.get_json()
    prompt = data.get('prompt')
    tenant_id = data.get('tenant_id')
    chat_mode = data.get('chat_mode', 'All-Purpose')

    print("Deleting screenshots")
    # clear png files
    delete_screenshots()

    inputs = {"question": prompt}
    print("Building ChatApp")
    chat_app = buildChatApp(tenant_id, chat_mode)

    try:
        for output in chat_app.stream(inputs):
            for key, value in output.items():
                print(f"Finished running: {key}:")
                pprint(value)
                print()
        
        return value["generation"]

    except Exception as e:
        print(f"Error location: {traceback.format_exc()}")
        print(f"Error: {e}")
    
    return f"I'm sorry, I'm unable to generate a response at the moment. Please try again later."

@app.route('/images', methods=['GET'])
def images_endpoint():
    # for all images in the output folder ending with .png, return the image.
    images = []
    for file in os.listdir("output"):
        if file.endswith(".png"):
            images.append(file)
    return {"images": images}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
