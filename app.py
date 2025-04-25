from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

app = Flask(__name__)

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location=device))
model.to(device)
model.eval()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    cleaned_text = clean_text(text)
    inputs = tokenizer(cleaned_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        output = model(**inputs).logits
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
    
    sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
    result = {
        'sentiment': sentiment_labels[prediction.item()],
        'confidence': confidence.item(),
        'text': text
    }
    return jsonify(result)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)