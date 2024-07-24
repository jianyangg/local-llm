import requests
import os

def upload_file(file_path, save_path, url):
    assert os.path.exists(file_path), f"File not found: {file_path}"
    with open(file_path, 'rb') as file:
        response = requests.post(url, files={'file': file}, data={'save_path': save_path})
    return response

if __name__ == '__main__':
    file_path = 'topic_models_cache/3a0324baf003b8e07544cca07a0fe26fb4d4abf03b8f3664435bce93348c0948/topic_model'
    save_path = '3a0324baf003b8e07544cca07a0fe26fb4d4abf03b8f3664435bce93348c0948/topic_model'
    url = 'http://127.0.0.1:5000/upload'  # Change this to your server's address if running remotely

    response = upload_file(file_path, save_path, url)
    print(response.json())
