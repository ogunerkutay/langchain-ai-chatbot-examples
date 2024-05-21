import json
import requests

url = "https://api.vectorshift.ai/api/pipelines/run"

headers = {
    "Api-Key": "your_api_key",
}

body = {
    "inputs": json.dumps({"Kullanici_Sorusu": "value"}),
    "pipeline_name": "ChatBotGPT",
    "username": "barhur",
}

response = requests.post(url, headers=headers, data=body)
response = response.json()