import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast
from huggingface_hub import login
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

if not HUGGING_FACE_TOKEN:
    raise ValueError("HUGGING_FACE_TOKEN is not set. Please configure it in the .env file.")
login(token=HUGGING_FACE_TOKEN)

model_name = "meta-llama/Llama-3.2-1B"

try:
    model = LlamaForCausalLM.from_pretrained(model_name)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    inputs = tokenizer(user_message, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, temperature=0.4, top_k=50)
    bot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
