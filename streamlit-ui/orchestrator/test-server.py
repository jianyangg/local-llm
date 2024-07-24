from flask import Flask, request, jsonify
import os

app = Flask(__name__)


@app.route('/upload_topic_model', methods=['POST'])
def upload_topic_model():
    UPLOAD_FOLDER = 'topic_models_cache'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if 'file' not in request.files or 'save_path' not in request.form:
        return jsonify({"error": "File or save path part missing"}), 400

    file = request.files['file']
    save_path = request.form['save_path']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Create the directories if they do not exist
    full_save_path = os.path.join(UPLOAD_FOLDER, save_path)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

    print(f"Upload path is {UPLOAD_FOLDER}")
    print(f"Save path is {save_path}")
    print(f"Saving file to {full_save_path}")
    file.save(full_save_path)
    return jsonify({"message": f"File saved successfully to {full_save_path}"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
