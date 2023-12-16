model_id = "sentence-transformers/all-MiniLM-L6-v2"
with(open("Keys/huggingface.txt", "r")) as f:
    hf_token =  f.read()

import requests

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()
