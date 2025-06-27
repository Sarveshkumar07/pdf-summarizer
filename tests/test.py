import requests

# HF_API_KEY 
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

sample_text = "Artificial Intelligence (AI) is transforming the world rapidly."

print("Sending request...")
response = requests.post(
    HF_API_URL,
    headers=headers,
    json={"inputs": sample_text},
    timeout=120
)
print("Response received")

print(response.status_code)
print(response.text)
