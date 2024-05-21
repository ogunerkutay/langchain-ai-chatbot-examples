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
from typing import Union

class ModelServiceClientSingleton:
    _instance = None

    def __new__(cls, credentials):
        if cls._instance is None:
            cls._instance = ModelServiceClient(credentials)
        return cls._instance

load_dotenv('../resources/.env') #use this to load the api key from .env files

def get_api_key(key_name):
    """Get environment variables on the environment."""
    api_key = os.environ.get(key_name)
    if api_key is None:
        raise ValueError("API key is not set.")
    return api_key

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

    #print("Desktop client Access Token:", credentials.token)
    #print("------------------------------------------")

    # Initialize the ModelServiceClient
    model_service_client = ModelServiceClient(credentials=credentials)

    #models = model_service_client.list_models()
    #print('Available base models:', [m.name for m in models])
    #print("------------------------------------------")

    #tuned_models = model_service_client.list_tuned_models()
    #print('My tuned models:', [m.name for m in tuned_models])
    #print("------------------------------------------")

    return credentials, model_service_client

# Set debug mode
set_debug(False)

# Retrieve the model information
#tuned_model = model_service_client.get_tuned_model(name="tunedModels/balangnbilgi1-sx9tl2gu34jn")

class GenAIRunnable(Runnable):
    def __init__(self, model):
        self.model = model

    def invoke(self, input: Union[str, "Runnables"], run_manager: Any) -> "Runnables":
        message_list = input.to_messages()

        credentials, model_service_client = load_creds_show_models()
        genai.configure(api_key="", credentials=credentials) #API key and credentials are mutually exclusive and while credentials have the necessary authorization keys dont, thats why api_key=""

        # Access the system message content and human message content
        system_message_content = message_list[0].content
        human_message_content = message_list[1].content

        # Print the contents
        #print("System Message Content:", system_message_content)
        #print("Human Message Content:", human_message_content)

        # Create Content objects for system and human messages
        combined_message = system_message_content + "\n" + human_message_content
        
        # Use GenAI model to generate response
        result = self.model.generate_content(combined_message, generation_config=generation_config, safety_settings=safety_settings)
        return result.text



# Initialize Google Generative AI models
#base models dont need credentials
#llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0.1, credentials=credentials, convert_system_message_to_human=True)
#tuned model need credentials
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

llm = genai.GenerativeModel(model_name="tunedModels/balangnbilgi1-sx9tl2gu34jn", generation_config=generation_config, safety_settings=safety_settings)
genai_runnable = GenAIRunnable(llm)

core_embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")

# Create a local file store for caching embeddings
store = LocalFileStore("./cache/")

# Setup embedder with cache
embedder = CacheBackedEmbeddings.from_bytes_store(core_embeddings_model, store, namespace=core_embeddings_model.model)

# Load and process documents for vector store creation
def load_and_process_document(file_path, chunk_size=2000, chunk_overlap=0):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Define the document directory based on the environment
document_directory = "source.txt"  # Update this path accordingly

if os.path.isfile(document_directory):
    document_texts = load_and_process_document(document_directory)
else:
    print(f"The document directory '{document_directory}' does not exist.")
    sys.exit(1)

# Create and populate the vector store
def create_vector_store(document_texts, embedder):
    if document_texts:
        if hasattr(embedder, 'embed_documents'):
            return FAISS.from_documents(document_texts, embedder)
        else:
            print("The embedder does not support document embedding.")
            return None
    else:
        print("No documents provided for vector store creation.")
        return None

vectorstore = create_vector_store(document_texts, embedder)
retriever = vectorstore.as_retriever()

system_template = """
Sen, IT ve makine bakımı konularında net ve açıklayıcı bilgiler sunan bir yapay zeka asistanısın. Adın MahmutAI.
Yanıtların açıklayıcı ve net olsun. Yanıtlarında güler yüz ve emoji kullanarak kullanıcı deneyimini iyileştir.
Birazdan sorulacak soru hakkında bilgi sahibi değilsen, "bilmiyorum" diyerek dürüst bir yanıt ver.
Eğer bir kaynakça gösterebiliyorsan kaynakçayı belirt ve link göster ama asla bağlamı sunma.
Sondaki soruyu yanıtlamak için aşağıdaki bağlamı kullan: {context}." 
"""

system_message = SystemMessagePromptTemplate.from_template(system_template)
human_message = HumanMessagePromptTemplate.from_template("{question}")
messages = [system_message, human_message]
prompt = ChatPromptTemplate.from_messages(messages)
#    {"context": retriever, "question": RunnablePassthrough(input_data="question")}

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | genai_runnable #or llm if ChatGoogleGenerativeAI used with base models
    | StrOutputParser()
)

def main():
    while True:
        query = input("Sorgu: ")
        if query.lower() == "exit":
            break
        response = chain.invoke(query)
        print(f"Yanıt: {response}")

if __name__ == "__main__":
    main()
