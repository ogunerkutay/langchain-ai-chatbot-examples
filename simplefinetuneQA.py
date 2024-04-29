import os
import sys
from tempfile import TemporaryDirectory
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.globals import set_debug
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.runnables.base import Runnable
from google.ai.generativelanguage import ModelServiceClient
from google.oauth2 import service_account
import google.generativeai as genai
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from typing import Any
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda
from google.generativeai.types import content_types

def load_creds():
    """Converts `client_secret.json` to a credential object.

    This function caches the generated tokens to minimize the use of the
    consent screen.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.

    # Define the OAuth 2.0 scopes required by your application
    SCOPES = ['https://www.googleapis.com/auth/cloud-platform',
          'https://www.googleapis.com/auth/generative-language.retriever',
          'https://www.googleapis.com/auth/generative-language.tuning']

    if os.path.exists('../resources/token.json'):
        creds = Credentials.from_authorized_user_file('../resources/token.json', SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Your OAuth2 client credentials file (JSON format) path
            CLIENT_SECRET_FILE = '../resources/client_secret.json'
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('../resources/token.json', 'w') as token:
            token.write(creds.to_json())

    if creds is None:
        raise ValueError("Credentials are not set.")

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../resources/client_secret.json'

    return creds

def load_creds_show_models():
    try:
        credentials = load_creds()
    except ValueError as e:
        print(f"Error loading credentials: {e}")
        sys.exit(1)    

    print("Desktop client Access Token:", credentials.token)
    print("------------------------------------------")

    # Initialize the ModelServiceClient
    model_service_client = ModelServiceClient(credentials=credentials)

    models = model_service_client.list_models()
    print('Available base models:', [m.name for m in models])
    print("------------------------------------------")

    tuned_models = model_service_client.list_tuned_models()
    print('My tuned models:', [m.name for m in tuned_models])
    print("------------------------------------------")

    return credentials, model_service_client


credentials, model_service_client = load_creds_show_models()


#genai.configure(api_key="exampleAPIKey")
genai.configure(api_key="",credentials=credentials)

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="tunedModels/balangnbilgi1-sx9tl2gu34jn", generation_config=generation_config, safety_settings=safety_settings)

prompt_parts = [
  "Explain the microorganisms in the story and their actions in a cause-effect relationship. ",
]

response = model.generate_content(prompt_parts)
print(response.text)