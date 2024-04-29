import os
from dotenv import load_dotenv
import requests


# Load environment variables
load_dotenv('../resources/.env')

def get_api_key(key_name):
    """Get environment variables on the environment."""
    return os.environ.get(key_name)

# API Keys
huggingfacehub_api_token = get_api_key("HUGGINGFACEHUB_API_TOKEN")


API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"
headers = {"Authorization": f"Bearer {huggingfacehub_api_token}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

data = query(
    {
	"inputs": "selam ",
}
)


if 'error' in data:
    print("Error: " + data['error'])
if 'warnings' in data:
    warnings = data['warnings']
    print("Warnings: " + ", ".join(map(str, warnings)))
elif 'generated_text' in data[0]:
    print("Generated Text: " + data[0]['generated_text'])
elif 'error' not in data and 'warnings' not in data and 'generated_text' not in data[0]:
    print("No error, warnings, or generated text found.")