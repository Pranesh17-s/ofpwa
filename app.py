from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import onnxruntime as ort
import os
from transformers import RobertaTokenizer
from deep_translator import GoogleTranslator
from langdetect import detect

app = Flask(__name__)
CORS(app)

# Load the ONNX model
onnx_model_path = 'C:/MobPwa/models/roberta_sentiment.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# Emotion labels based on model output
emotion_labels = {
    0: "Negative 😕",
    1: "Neutral 😐",
    2: "Positive 🙂"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    user_input = data.get('text', '')

    # Detect language and translate to English if necessary
    detected_language = detect(user_input)
    if detected_language != 'en':
        user_input = GoogleTranslator(source=detected_language, target='en').translate(user_input)

    # Preprocess the input text
    inputs = preprocess_text(user_input)

    # Run inference using the ONNX model
    inputs_onnx = {
        "input_ids": inputs['input_ids'], 
        "attention_mask": inputs['attention_mask'], 
        "token_type_ids": inputs['token_type_ids']
    }
    outputs = ort_session.run(None, inputs_onnx)

    # Get the predicted class
    predicted_class = np.argmax(outputs[0], axis=1)[0]
    emotion = emotion_labels.get(predicted_class, "Unknown")

    return jsonify({
        "predicted_sentiment": emotion  # Return only the sentiment
    })

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('.', 'manifest.json')

# Serve the ONNX model file from the /models directory
@app.route('/models/<path:filename>')
def serve_model(filename):
    return send_from_directory(os.path.join(app.root_path, 'models'), filename)

def preprocess_text(text):
    # Tokenize the input text and return the necessary inputs for the ONNX model
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=512)

    return {
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask'],
        "token_type_ids": inputs.get('token_type_ids', np.zeros_like(inputs['input_ids']))  # Use zeros if not present
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7070))
    app.run(debug=True, host='0.0.0.0', port=port)
